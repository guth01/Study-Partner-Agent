"""
Agent tools — the actual capabilities the agent uses.

Tools are plain functions called by LangGraph nodes.
They are NOT LangChain tool objects — just clean Python functions.

Day 5 additions:
  create_study_session_event  — create a Google Calendar event
  get_upcoming_events         — list next 10 calendar events
  generate_study_plan         — propose study sessions from gap analysis
  create_flashcards           — generate + persist SM-2 flashcards
  generate_exam_revision_sheet — full revision document per subject
  translate_content           — HuggingFace NLLB translation

Note: fetch_wikipedia_summary has been removed.
The Wikipedia fallback is replaced by the Sufficiency Judge + Tavily pipeline.
"""

import os
import json
import asyncio
import requests
from typing import Optional, List
from datetime import datetime, timedelta, timezone
import re

from db.chroma import get_session_collection
from utils.embedder import get_embeddings


# ============================================================================
# TOOL: search_notes
# ============================================================================

def search_notes(query: str, session_id: str, top_k: int = 3) -> dict:
    """
    Search the session's ChromaDB collection for relevant notes.
    Automatically translates the query to English before searching since the embedding model is English-optimized.

    Args:
        query: User's query (in any language)
        session_id: Active session ID
        top_k: Number of chunks to retrieve

    Returns:
        dict with chunks, confidence score, and found boolean
    """
    try:
        # Detect language to avoid translating English queries
        from langdetect import detect
        try:
            lang = detect(query)
        except Exception:
            lang = "en"  # fallback if detection fails

        if lang != "en":
            # Translate query to English for optimal embedding match against English notes
            translated = translate_content(query, "EN")
            search_query = translated.get("translated_text", query)
            print(f"[TOOL:search_notes] Original query: '{query}'")
            print(f"[TOOL:search_notes] Translated query: '{search_query}'")
        else:
            search_query = query

        from langchain.retrievers import ContextualCompressionRetriever
        from langchain_cohere import CohereRerank

        vs = get_session_collection(session_id)

        # Stage 1: Retrieve a larger pool of documents (fast similarity search)
        initial_k = max(20, top_k * 5)
        base_retriever = vs.as_retriever(search_kwargs={"k": initial_k})
        
        # Stage 2: Rerank the retrieved documents using Cohere
        compressor = CohereRerank(top_n=top_k)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )

        # invoke returns a list of Document objects sorted by rerank score
        compressed_docs = compression_retriever.invoke(search_query)

        if not compressed_docs:
            return {"chunks": [], "confidence": 0.0, "found": False}

        chunks = []
        for doc in compressed_docs:
            # LangChain CohereRerank stores the rerank score in metadata
            score = doc.metadata.get("relevance_score", 0.0)
            chunks.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": round(float(score), 4),
            })

        # Confidence = score of the single best matching chunk
        # (Averaging top-3 penalizes specific queries where only 1 chunk is relevant)
        confidence = chunks[0]["score"] if chunks else 0.0

        print(f"[TOOL:search_notes] query='{query}' -> {len(chunks)} chunks, confidence={confidence:.3f}")
        return {
            "chunks": chunks,
            "confidence": round(confidence, 4),
            "found": len(chunks) > 0,
        }

    except ValueError as e:
        # ChromaDB collection doesn't exist (session ended / invalid)
        raise RuntimeError(f"Notes database error: {e}")

    except Exception as e:
        raise RuntimeError(f"Unexpected error searching notes: {e}")


# ============================================================================
# TOOL: summarize_topic  (calls search_notes + Gemini)
# ============================================================================

DEPTH_PROMPTS = {
    "quick": "Give a brief 2-3 sentence overview suitable for a quick refresher.",
    "detailed": "Give a thorough explanation covering the main concepts, mechanisms, and examples.",
    "exam_revision": "Structure your response for exam revision: key definitions, important points, common exam questions, and memory tips.",
    "beginner": "Explain this as if to someone with no background. Use simple language, analogies, and avoid jargon.",
    "advanced": "Provide a deep, technical explanation assuming graduate-level background. Include edge cases, nuances, and connections to related concepts.",
}


async def summarize_topic(
    topic: str,
    session_id: str,
    depth_level: str = "detailed",
    llm=None,
) -> dict:
    """
    Summarize a topic using the session's notes via RAG.

    Args:
        topic:       Topic to summarize
        session_id:  Active session ID for RAG
        depth_level: quick | detailed | exam_revision | beginner | advanced
        llm:         Gemini LLM instance (passed in from nodes)

    Returns:
        {
            "topic": str,
            "summary": str,
            "depth_level": str,
            "source_chunks_used": int
        }
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    if llm is None:
        raise ValueError("summarize_topic requires an LLM instance")

    # Step 1: Retrieve notes
    notes_result = search_notes(topic, session_id, top_k=6)
    chunks = notes_result.get("chunks", [])

    # Build context from notes
    if chunks:
        notes_context = "\n\n---\n\n".join([c["content"] for c in chunks])
        source_info = f"(from {len(chunks)} chunks in your notes)"
    else:
        notes_context = "No relevant notes found for this topic."
        source_info = "(no notes found — using general knowledge)"

    depth_instruction = DEPTH_PROMPTS.get(depth_level, DEPTH_PROMPTS["detailed"])

    system_prompt = f"""You are a focused study assistant. Your task is to summarize a topic based on the student's own notes.

{depth_instruction}

Use ONLY the provided notes as your primary source. If notes are insufficient, supplement with your knowledge but clearly indicate this.
Respond in the same language that the user's notes and topic are in."""

    user_prompt = f"""Topic: {topic}

Student's Notes:
{notes_context}

Please provide a {depth_level} summary of "{topic}" {source_info}."""

    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    summary_text = response.content if hasattr(response, "content") else str(response)

    print(f"[TOOL:summarize_topic] topic='{topic}', depth={depth_level}, chunks={len(chunks)}")
    return {
        "topic": topic,
        "summary": summary_text,
        "depth_level": depth_level,
        "source_chunks_used": len(chunks),
    }


# ============================================================================
# TOOL: knowledge_gap_analysis
# ============================================================================


# Thresholds
WELL_COVERED_THRESHOLD = 0.55    # confidence >= this → well covered
SHALLOW_THRESHOLD = 0.30         # confidence between shallow and well_covered → shallow


def knowledge_gap_analysis(
    session_id: str,
    subject_name: str = "default",
    custom_topics: Optional[list] = None,
) -> dict:
    """
    Analyze which topics are well-covered, shallow, or missing in the session's notes.

    Args:
        session_id:    Active session ID for RAG
        subject_name:  Subject name to pick default topics (e.g. "computer science")
        custom_topics: Override with a specific list of topics to check

    Returns:
        {
            "well_covered": [str],
            "shallow": [str],
            "missing": [str],
            "details": {topic: {"confidence": float, "chunk_count": int}}
        }
    """
    # Determine topic list
    if custom_topics and len(custom_topics) > 0:
        topics = custom_topics
    else:
        topics = ["General Review"]

    well_covered = []
    shallow = []
    missing = []
    details = {}

    print(f"[TOOL:gap_analysis] Analyzing {len(topics)} topics for session {session_id}")

    for topic in topics:
        result = search_notes(topic, session_id, top_k=5)
        confidence = result.get("confidence", 0.0)
        chunk_count = len(result.get("chunks", []))

        details[topic] = {"confidence": confidence, "chunk_count": chunk_count}

        if confidence >= WELL_COVERED_THRESHOLD:
            well_covered.append(topic)
        elif confidence >= SHALLOW_THRESHOLD:
            shallow.append(topic)
        else:
            missing.append(topic)

    print(f"[TOOL:gap_analysis] DONE covered={len(well_covered)}, shallow={len(shallow)}, missing={len(missing)}")
    return {
        "well_covered": well_covered,
        "shallow": shallow,
        "missing": missing,
        "details": details,
    }


# ============================================================================
# TOOL: create_study_session_event
# ============================================================================

async def create_study_session_event(
    user_id: str,
    subject: str,
    date: str,              # ISO format: "2025-04-10"
    duration_minutes: int,
    db,                     # AsyncIOMotorDatabase
    start_hour: int = 9,    # Default start time: 9 AM
    start_time: str = None, # "HH:MM" 24-hour format
    topic: str = None,      # Optional topic label for event title
) -> dict:
    """
    Create a Google Calendar event for a study session.

    Args:
        user_id:          Atlas user ID
        subject:          Subject name (used as event title)
        date:             Date string in YYYY-MM-DD format
        duration_minutes: Session duration
        db:               AsyncIOMotorDatabase instance
        start_hour:       Hour of day to start (24h, default 9)

    Returns:
        {
            "event_id": str,
            "html_link": str,
            "title": str,
            "start": str,
            "end": str
        }
    """
    from utils.google_calendar import get_calendar_service

    try:
        service = await get_calendar_service(user_id, db)
        loop = asyncio.get_event_loop()

        # Get user's actual timezone from their calendar
        calendar_info = await loop.run_in_executor(
            None,
            lambda: service.calendars().get(calendarId="primary").execute()
        )
        user_tz_str = calendar_info.get("timeZone", "UTC")

        # Build naive datetimes
        if start_time:
            hh, mm = map(int, start_time.split(":"))
            start_dt = datetime.strptime(date, "%Y-%m-%d").replace(
                hour=hh, minute=mm, second=0
            )
        else:
            start_dt = datetime.strptime(date, "%Y-%m-%d").replace(
                hour=start_hour, minute=0, second=0
            )
        
        end_dt = start_dt + timedelta(minutes=duration_minutes)

        event_body = {
            "summary": f"📚 Study: {subject} - {topic}" if topic else f"📚 Study: {subject}",
            "description": (
                f"Study session for {subject}\n"
                f"Duration: {duration_minutes} minutes\n"
                f"Created by Study Agent"
            ),
            "start": {
                "dateTime": start_dt.isoformat(),
                "timeZone": user_tz_str,
            },
            "end": {
                "dateTime": end_dt.isoformat(),
                "timeZone": user_tz_str,
            },
            "colorId": "2",      # Sage green — matches study theme
            "reminders": {
                "useDefault": False,
                "overrides": [
                    {"method": "popup", "minutes": 15},
                ],
            },
        }

        # Run the blocking Google API call in a thread pool
        loop = asyncio.get_event_loop()
        event = await loop.run_in_executor(
            None,
            lambda: service.events().insert(
                calendarId="primary", body=event_body
            ).execute()
        )

        print(f"[TOOL:create_event] Created '{event_body['summary']}' on {date}")
        return {
            "event_id": event.get("id", ""),
            "html_link": event.get("htmlLink", ""),
            "title": event_body["summary"],
            "start": start_dt.isoformat(),
            "end": end_dt.isoformat(),
        }

    except ValueError as e:
        # No Google auth linked
        print(f"[TOOL:create_event] Auth error: {e}")
        return {"error": str(e), "html_link": "", "title": subject}
    except Exception as e:
        print(f"[TOOL:create_event] Error: {e}")
        return {"error": str(e), "html_link": "", "title": subject}


# ============================================================================
# TOOL: get_upcoming_events
# ============================================================================

async def get_upcoming_events(user_id: str, db) -> list:
    """
    Fetch the next 10 upcoming Google Calendar events for the user.

    Args:
        user_id: Atlas user ID
        db:      AsyncIOMotorDatabase instance

    Returns:
        List of dicts: [{"title": str, "start": str, "end": str, "html_link": str}]
    """
    from utils.google_calendar import get_calendar_service

    try:
        service = await get_calendar_service(user_id, db)

        now = datetime.now(tz=timezone.utc).isoformat()

        loop = asyncio.get_event_loop()
        events_result = await loop.run_in_executor(
            None,
            lambda: service.events().list(
                calendarId="primary",
                timeMin=now,
                maxResults=10,
                singleEvents=True,
                orderBy="startTime",
            ).execute()
        )

        items = events_result.get("items", [])
        events = []
        for item in items:
            start = item["start"].get("dateTime", item["start"].get("date", ""))
            end = item["end"].get("dateTime", item["end"].get("date", ""))
            events.append({
                "title": item.get("summary", "(No title)"),
                "start": start,
                "end": end,
                "html_link": item.get("htmlLink", ""),
            })

        print(f"[TOOL:get_events] Found {len(events)} upcoming events")
        return events

    except ValueError as e:
        print(f"[TOOL:get_events] Auth error: {e}")
        return [{"error": str(e)}]
    except Exception as e:
        print(f"[TOOL:get_events] Error: {e}")
        return [{"error": str(e)}]


# ============================================================================
# TOOL: generate_study_plan
# ============================================================================

def generate_study_plan(
    session_id: str,
    subject_name: str,
    exam_date: str,                    # "YYYY-MM-DD" or natural like "April 15"
    session_duration_minutes: int = 60,
    custom_topics: Optional[List[str]] = None,
) -> dict:
    """
    Generate a proposed study plan based on gap analysis.

    Weights topics by coverage:
      - missing  → 3 sessions
      - shallow  → 2 sessions
      - well_covered → 1 session (for review)

    Distributes sessions from tomorrow to exam_date (weekends included by
    default — user can ask to skip weekends in their message).

    Args:
        session_id:               Active session ID for gap analysis
        subject_name:             Subject name (e.g. "Operating Systems")
        exam_date:                Exam date string
        session_duration_minutes: Duration per study session in minutes
        custom_topics:            Optional list of topics to override defaults

    Returns:
        {
            "proposed_events": [{"subject": str, "topic": str, "date": str, "duration_minutes": int}],
            "gap_analysis": {...},
            "total_sessions": int
        }
    """
    # Parse exam_date — try ISO first, then natural language
    try:
        exam_dt = datetime.strptime(exam_date, "%Y-%m-%d")
    except ValueError:
        # Try natural formats like "April 15", "15 April 2025", etc.
        for fmt in ("%B %d", "%d %B", "%B %d %Y", "%d %B %Y"):
            try:
                parsed = datetime.strptime(exam_date, fmt)
                # If no year, assume next occurrence
                exam_dt = parsed.replace(year=datetime.now().year)
                if exam_dt < datetime.now():
                    exam_dt = exam_dt.replace(year=datetime.now().year + 1)
                break
            except ValueError:
                continue
        else:
            # Fallback: 14 days from now
            exam_dt = datetime.now() + timedelta(days=14)
            print(f"[TOOL:study_plan] Could not parse exam_date='{exam_date}', defaulting to 14 days")

    # Run gap analysis
    gap = knowledge_gap_analysis(
        session_id=session_id,
        subject_name=subject_name,
        custom_topics=custom_topics,
    )

    # Build weighted topic list: missing x3, shallow x2, covered x1
    weighted_topics = []
    for topic in gap.get("missing", []):
        weighted_topics.extend([(topic, "missing")] * 3)
    for topic in gap.get("shallow", []):
        weighted_topics.extend([(topic, "shallow")] * 2)
    for topic in gap.get("well_covered", []):
        weighted_topics.extend([(topic, "well_covered")] * 1)

    if not weighted_topics:
        # Fallback if no topics found
        weighted_topics = [("General Review", "missing")] * 3

    # Build date range: tomorrow → day before exam
    today = datetime.now().date()
    start_date = today + timedelta(days=1)
    end_date = exam_dt.date() - timedelta(days=1)  # stop before exam day

    available_dates = []
    current = start_date
    while current <= end_date:
        available_dates.append(current)
        current += timedelta(days=1)

    if not available_dates:
        # Exam is tomorrow or already passed — just add today
        available_dates = [today]

    # Distribute sessions across available dates
    proposed_events = []
    for i, (topic, coverage) in enumerate(weighted_topics):
        date = available_dates[i % len(available_dates)]
        proposed_events.append({
            "subject": subject_name,
            "topic": topic,
            "date": date.strftime("%Y-%m-%d"),
            "duration_minutes": session_duration_minutes,
            "coverage_level": coverage,
        })

    # Sort by date
    proposed_events.sort(key=lambda e: e["date"])

    print(f"[TOOL:study_plan] Generated {len(proposed_events)} sessions for '{subject_name}'")
    return {
        "proposed_events": proposed_events,
        "gap_analysis": gap,
        "total_sessions": len(proposed_events),
        "exam_date": exam_dt.strftime("%Y-%m-%d"),
    }


# ============================================================================
# TOOL: create_flashcards
# ============================================================================

async def create_flashcards(
    topic: str,
    num_cards: int,
    session_id: str,
    subject_id: str,
    user_id: str,
    db,                      # AsyncIOMotorDatabase
    llm=None,                # Gemini LLM instance (passed from node)
) -> dict:
    """
    Generate flashcards on a topic using RAG + Gemini, then persist to Atlas.

    Each card gets an initial status:
        status: "upcoming"

    Args:
        topic:      Topic to generate flashcards for
        num_cards:  Number of cards to generate
        session_id: Active session ID for RAG context
        subject_id: Subject ID for Atlas storage
        user_id:    User ID for Atlas storage
        db:         AsyncIOMotorDatabase instance
        llm:        Gemini LLM instance

    Returns:
        {
            "cards_created": int,
            "cards": [{"question": str, "answer": str, "card_type": str}],
            "message": str
        }
    """
    from langchain_core.messages import HumanMessage, SystemMessage
    from bson import ObjectId

    if llm is None:
        raise ValueError("create_flashcards requires an LLM instance")

    # Step 1: Get context from notes
    notes_result = search_notes(topic, session_id, top_k=6)
    chunks = notes_result.get("chunks", [])

    if chunks:
        notes_context = "\n\n---\n\n".join([c["content"] for c in chunks[:4]])
    else:
        notes_context = f"No specific notes found for '{topic}'. Use general knowledge."

    # Step 2: Prompt Gemini for structured flashcard JSON
    system_prompt = """You are an expert flashcard creator for students.
Generate exactly the requested number of high-quality flashcards.

Return ONLY a valid JSON array with no markdown, no code blocks, no extra text.
Each object in the array must have exactly these fields:
  - "question": clear, specific question
  - "answer": concise but complete answer
  - "card_type": one of "definition", "concept", "application", "fact"

Generate the flashcards in the same language as the provided topic and context.
Ensure all generated questions are strictly unique and diverse. Do NOT repeat the same or similar questions.

Example format:
[{"question": "What is...", "answer": "It is...", "card_type": "definition"}]"""

    user_prompt = f"""Topic: {topic}
Number of cards: {num_cards}

Context from student notes:
{notes_context}

Generate {num_cards} flashcards on "{topic}" based on the notes above."""

    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    raw_content = response.content if hasattr(response, "content") else str(response)

    # Step 3: Parse JSON (handle wrapped/dirty output)
    try:
        # Strip markdown code blocks if present
        clean = re.sub(r"```(?:json)?\n?", "", raw_content).strip().rstrip("`")
        cards_data = json.loads(clean)
        if not isinstance(cards_data, list):
            raise ValueError("Expected a JSON array")
    except (json.JSONDecodeError, ValueError) as e:
        raise RuntimeError(f"Failed to generate flashcards (invalid formatting from LLM): {e}")

    # Step 4: Persist to Atlas with SM-2 initial values
    tomorrow = datetime.utcnow() + timedelta(days=1)
    now = datetime.utcnow()

    card_docs = []
    for card in cards_data:
        doc = {
            "user_id": user_id,
            "session_id": session_id,
            "subject_id": subject_id,
            "topic": topic,
            "question": card.get("question", ""),
            "answer": card.get("answer", ""),
            "card_type": card.get("card_type", "concept"),
            "created_at": now,
            "status": "upcoming",
        }
        card_docs.append(doc)

    if card_docs:
        await db.flashcards.insert_many(card_docs)

    print(f"[TOOL:flashcards] Created {len(card_docs)} flashcards on '{topic}'")

    # Step 5: Return human-readable format
    readable_cards = [
        {
            "question": c["question"],
            "answer": c["answer"],
            "card_type": c["card_type"],
        }
        for c in card_docs
    ]

    return {
        "cards_created": len(card_docs),
        "cards": readable_cards,
        "topic": topic,
        "message": f"✅ Created {len(card_docs)} flashcards on '{topic}' and saved to your collection.",
    }


# ============================================================================
# TOOL: generate_exam_revision_sheet
# ============================================================================

async def generate_exam_revision_sheet(
    subject_id: str,
    session_id: str,
    subject_name: str,
    llm=None,
    custom_topics: Optional[List[str]] = None,
) -> dict:
    """
    Generate a comprehensive exam revision sheet for a subject.

    Process:
    1. Run gap_analysis to identify all topics
    2. Call summarize_topic (exam_revision depth) for each topic
    3. Compile a structured markdown document

    Args:
        subject_id:   Subject ID for context
        session_id:   Active session ID for RAG
        subject_name: Subject name for display
        llm:          Gemini LLM instance

    Returns:
        {
            "document": str,   # Full markdown revision sheet
            "topics_covered": int,
            "subject": str
        }
    """
    if llm is None:
        raise ValueError("generate_exam_revision_sheet requires an LLM instance")

    print(f"[TOOL:revision_sheet] Generating revision sheet for '{subject_name}'")

    # Step 1: Gap analysis to get all topics
    gap = knowledge_gap_analysis(
        session_id=session_id,
        subject_name=subject_name,
        custom_topics=custom_topics,
    )

    all_topics = (
        gap.get("well_covered", [])
        + gap.get("shallow", [])
        + gap.get("missing", [])
    )

    if not all_topics:
        return {
            "document": f"# {subject_name} Revision Sheet\n\nNo topics found in your notes.",
            "topics_covered": 0,
            "subject": subject_name,
        }

    # Step 2: Build revision sheet sections
    now_str = datetime.utcnow().strftime("%B %d, %Y")
    sections = [
        f"# 📖 {subject_name} — Exam Revision Sheet",
        f"*Generated: {now_str}*\n",
        "---\n",
        "## Knowledge Coverage Summary",
        f"- ✅ Well covered ({len(gap.get('well_covered', []))} topics): "
        f"{', '.join(gap.get('well_covered', [])) or 'none'}",
        f"- ⚠️ Needs work ({len(gap.get('shallow', []))} topics): "
        f"{', '.join(gap.get('shallow', [])) or 'none'}",
        f"- ❌ Missing ({len(gap.get('missing', []))} topics): "
        f"{', '.join(gap.get('missing', [])) or 'none'}",
        "\n---\n",
    ]

    # Step 3: Summarize each topic at exam_revision depth
    # Cap at 3 topics and sleep 2s between calls to stay safely under the limit.
    capped_topics = all_topics[:3]
    print(f"[TOOL:revision_sheet] Summarizing {len(capped_topics)} topics (capped to 3 for rate limits)")

    for i, topic in enumerate(capped_topics):
        if i > 0:
            await asyncio.sleep(2)

        print(f"[TOOL:revision_sheet] Summarizing topic: '{topic}'")
        try:
            summary_result = await summarize_topic(
                topic=topic,
                session_id=session_id,
                depth_level="exam_revision",
                llm=llm,
            )
            summary_text = summary_result.get("summary", "No summary available.")

            # Coverage badge
            if topic in gap.get("well_covered", []):
                badge = "✅"
            elif topic in gap.get("shallow", []):
                badge = "⚠️"
            else:
                badge = "❌"

            sections.append(f"## {badge} {topic.title()}")
            sections.append(summary_text)
            sections.append("\n---\n")

        except Exception as e:
            print(f"[TOOL:revision_sheet] Error summarizing '{topic}': {e}")
            sections.append(f"## {topic.title()}")
            sections.append(f"*Error generating summary: {e}*")
            sections.append("\n---\n")

    document = "\n".join(sections)

    print(f"[TOOL:revision_sheet] Done. {len(all_topics)} topics, {len(document)} chars")
    return {
        "document": document,
        "topics_covered": len(all_topics),
        "subject": subject_name,
    }


# ============================================================================
# TOOL: translate_content
# ============================================================================

def translate_content(text: str, target_language: str) -> dict:
    """
    Translate text using the Hugging Face Inference API (NLLB-200).

    Args:
        text:            Text to translate
        target_language: Target language code (e.g. "DE", "FR", "ES", "ZH")

    Returns:
        {
            "translated_text": str,
            "source_language": str,
            "target_language": str
        }
        or
        {
            "error": str,
            "original_text": str
        }
    """
    api_key = os.getenv("HF_TOKEN", "")

    if not api_key:
        print("[TOOL:translate] HF_TOKEN not configured")
        return {
            "error": "HF_TOKEN not configured. Set HF_TOKEN in .env",
            "original_text": text,
        }

    # Map common short codes to NLLB-200 language codes
    nllb_codes = {
        "FR": "fra_Latn", "DE": "deu_Latn", "ES": "spa_Latn",
        "ZH": "zho_Hans", "EN": "eng_Latn", "IT": "ita_Latn",
        "PT": "por_Latn", "RU": "rus_Cyrl", "JA": "jpn_Jpan",
        "AR": "arb_Arab", "HI": "hin_Deva"
    }

    lang_code = target_language.strip().upper()
    tgt_lang = nllb_codes.get(lang_code, "eng_Latn")
    
    if lang_code not in nllb_codes and len(target_language) > 3:
        tgt_lang = target_language

    try:
        url = "https://api-inference.huggingface.co/models/facebook/nllb-200-distilled-600M"
        headers = {"Authorization": f"Bearer {api_key}"}
        payload = {
            "inputs": text,
            "parameters": {"tgt_lang": tgt_lang}
        }

        response = requests.post(url, headers=headers, json=payload, timeout=15)
        response.raise_for_status()

        result = response.json()
        
        if isinstance(result, list) and len(result) > 0 and "translation_text" in result[0]:
            translated_text = result[0]["translation_text"]
        else:
            translated_text = str(result)

        print(f"[TOOL:translate] translated to {tgt_lang}, {len(text)} chars")
        return {
            "translated_text": translated_text,
            "source_language": "AUTO",
            "target_language": lang_code,
        }

    except Exception as e:
        print(f"[TOOL:translate] HF API error: {e}")
        return {
            "error": f"Translation failed: {str(e)}",
            "original_text": text,
        }
