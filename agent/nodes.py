"""
LangGraph nodes — each node transforms the AgentState.

Every node is a plain async function that takes state and returns
a dict of state fields to update (LangGraph merges them).

LLM used: Gemini (via langchain-google-genai) for judge + synthesis.
           Groq for fast classification tasks (router, planner, etc.)

RAG Pipeline (updated — Sufficiency Judge replaces Wikipedia fallback):
  rag_node
    └── sufficiency_judge_node (Gemini key 0 — judge key)
          ├── SUFFICIENT  →  synthesis_node (Gemini key 1 — answer key)
          └── PARTIAL / INSUFFICIENT → await_fallback_node [interrupt]
                ├── gemini strategy  →  synthesis_node
                └── tavily strategy  →  tavily_search_node → synthesis_node

Day 5 additions (unchanged):
  gap_analysis_node       — runs knowledge_gap_analysis, stores result in state
  study_plan_builder_node — calls generate_study_plan, stores proposed events
  calendar_node           — creates Google Calendar events for each proposed session
  flashcard_generator_node — calls create_flashcards, stores result in state
  revision_sheet_node     — calls generate_exam_revision_sheet, stores document

Routing additions:
  route_by_content_type   — after planner: flashcards/revision vs generic synthesis
  route_by_calendar_auth  — checks if user has Google OAuth tokens in Atlas
  Updated route_by_intent — study_planning now routes to gap_analysis (not planner)
"""

import os
import re
from typing import Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from agent.state import AgentState
from agent.tools import (
    search_notes,
    knowledge_gap_analysis,
    generate_study_plan,
    create_flashcards,
    generate_exam_revision_sheet,
    create_study_session_event,
    translate_content,
)
from agent.sufficiency_judge import run_sufficiency_judge
from agent.tavily_search import tavily_search, format_web_results_for_prompt


# ============================================================================
# LLM Factories — Two dedicated Gemini keys + Groq
# ============================================================================

import itertools

_answer_key: str | None = None  # Key index 1 — reserved for answer synthesis
_groq_llm_instance = None


def _get_judge_llm(temperature: float = 0.0) -> ChatGoogleGenerativeAI:
    """
    Get a Gemini LLM instance using GEMINI_KEYS[0].
    Reserved exclusively for the Sufficiency Judge.
    Kept here for any node that needs to call the judge LLM directly —
    the primary judge logic lives in agent/sufficiency_judge.py.
    """
    keys_str = os.getenv("GEMINI_API_KEYS", "")
    if keys_str:
        keys = [k.strip() for k in keys_str.split(",") if k.strip()]
        key = keys[0]
    else:
        key = os.getenv("GEMINI_API_KEY", "")

    if not key:
        raise RuntimeError("GEMINI_API_KEY or GEMINI_API_KEYS not set in environment")

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=key,
        temperature=temperature,
    )


def _get_answer_llm(temperature: float = 0.4) -> ChatGoogleGenerativeAI:
    """
    Get a Gemini LLM instance using GEMINI_KEYS[1].
    Reserved exclusively for answer synthesis (all 5 answer generation cases).
    Falls back to key[0] if only one key is configured.
    """
    global _answer_key

    if _answer_key is None:
        keys_str = os.getenv("GEMINI_API_KEYS", "")
        if keys_str:
            keys = [k.strip() for k in keys_str.split(",") if k.strip()]
            # Use key[1] if available, otherwise fall back to key[0]
            _answer_key = keys[1] if len(keys) > 1 else keys[0]
        else:
            single = os.getenv("GEMINI_API_KEY", "")
            if not single:
                raise RuntimeError("GEMINI_API_KEY or GEMINI_API_KEYS not set in environment")
            _answer_key = single

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=_answer_key,
        temperature=temperature,
        streaming=True,
    )


# Legacy rotating LLM — kept for revision_sheet_node which iterates over many topics
_api_key_cycle = None


def _get_llm(temperature: float = 0.3) -> ChatGoogleGenerativeAI:
    """Get a configured Gemini LLM instance, rotating through keys if multiple are provided."""
    global _api_key_cycle

    if _api_key_cycle is None:
        keys_str = os.getenv("GEMINI_API_KEYS")
        if keys_str:
            keys = [k.strip() for k in keys_str.split(",") if k.strip()]
        else:
            single_key = os.getenv("GEMINI_API_KEY")
            keys = [single_key] if single_key else []

        if not keys:
            raise RuntimeError("GEMINI_API_KEY or GEMINI_API_KEYS not set in environment")

        _api_key_cycle = itertools.cycle(keys)

    api_key = next(_api_key_cycle)

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=temperature,
    )


def _get_groq_llm(temperature: float = 0.3) -> ChatGroq:
    """Get a configured Groq LLM instance for fast classification tasks."""
    return ChatGroq(
        model="llama-3.3-70b-specdec",
        temperature=temperature,
    )


# ============================================================================
# RouterNode — classifies the user's intent
# ============================================================================

ROUTER_SYSTEM_PROMPT = """You are an intent classifier for a student study assistant.

The user might speak in other languages (e.g., Spanish, French). Classify the intent correctly regardless of the language they use.

Classify the user's message into exactly ONE of these intents:
- rag_query       → asking about content from their notes/documents (explain, what is, how does, etc.)
- content_generation → asking to create study material (flashcards, quiz, summary, mind map, revision sheet, etc.)
- study_planning  → asking about schedules, study plans, what to study, time management, exam prep, or knowledge gap analysis (e.g. what am I missing, analyze my notes for gaps)
- calendar_scheduling → specifically asking to schedule a single calendar event or study block at a specific time (e.g. schedule a 2 hour block tomorrow for OS)
- session_end     → wants to end the session (goodbye, done, quit, stop, finish)
- chitchat        → casual conversation, greetings, unrelated to studying
- translation     → asking to translate text to a specific language

Respond with ONLY the intent label, nothing else. No punctuation, no explanation.
Examples:
  User: "explain virtual memory" → rag_query
  User: "make me 10 flashcards on recursion" → content_generation
  User: "create a study plan for my exam" → study_planning
  User: "what topics am I missing in my notes?" → study_planning
  User: "plan my OS revision, exam is April 15" → study_planning
  User: "schedule a 90 minute block for databases tomorrow at 3pm" → calendar_scheduling
  User: "generate a revision sheet for databases" → content_generation
  User: "translate this to French: The study agent is working perfectly!" → translation
  User: "hi there!" → chitchat
  User: "I'm done for today" → session_end"""


async def router_node(state: AgentState) -> dict:
    """
    RouterNode: Classify the latest user message into an intent.
    Fast, cheap call — no context needed, just the last message.
    """
    print("[NODE:router] Classifying intent...")

    llm = _get_groq_llm(temperature=0.0)  # deterministic classification

    last_message = state["messages"][-1]["content"] if state["messages"] else ""

    response = await llm.ainvoke([
        SystemMessage(content=ROUTER_SYSTEM_PROMPT),
        HumanMessage(content=last_message),
    ])

    raw_intent = response.content.strip().lower()

    # Validate — fall back to rag_query if model hallucinates
    valid_intents = {"rag_query", "content_generation", "study_planning", "calendar_scheduling", "session_end", "chitchat", "translation"}
    intent = raw_intent if raw_intent in valid_intents else "rag_query"

    print(f"[NODE:router] intent='{intent}' (raw='{raw_intent}')")
    return {"intent": intent}


# ============================================================================
# RAGNode — semantic search over session's ChromaDB
# ============================================================================

async def rag_node(state: AgentState) -> dict:
    """
    RAGNode: Perform semantic search using the user's notes.
    Sets retrieved_chunks and chunk_confidence in state.
    Pipeline continues to sufficiency_judge_node regardless of confidence.
    """
    print("[NODE:rag] Searching notes...")

    last_message = state["messages"][-1]["content"] if state["messages"] else ""
    session_id = state["session_id"]

    result = search_notes(query=last_message, session_id=session_id, top_k=5)

    chunks = result.get("chunks", [])
    confidence = result.get("confidence", 0.0)

    print(f"[NODE:rag] Retrieved {len(chunks)} chunks, confidence={confidence:.3f}")
    return {
        "retrieved_chunks": chunks,
        "chunk_confidence": confidence,
    }


# ============================================================================
# SufficiencyJudgeNode — evaluate whether retrieved chunks are adequate
# ============================================================================

async def sufficiency_judge_node(state: AgentState) -> dict:
    """
    SufficiencyJudgeNode: Call the Sufficiency Judge (Gemini key 0) to classify
    whether the retrieved notes can answer the user's question.

    Sets judge_verdict and judge_reason in state.

    Logs:
        [NODE:sufficiency_judge] question
        [NODE:sufficiency_judge] chunks_count
        [NODE:sufficiency_judge] verdict
        [NODE:sufficiency_judge] reason
    """
    print("[NODE:sufficiency_judge] Evaluating retrieval quality...")

    last_message = state["messages"][-1]["content"] if state["messages"] else ""
    chunks = state.get("retrieved_chunks") or []

    # Format chunks into a single context string for the judge
    if chunks:
        retrieved_context = "\n\n---\n\n".join([
            f"[From: {c['metadata'].get('filename', 'notes')}]\n{c['content']}"
            for c in chunks
        ])
    else:
        retrieved_context = "No relevant content was retrieved from the notes."

    print(f"[NODE:sufficiency_judge] question='{last_message[:100]}'")
    print(f"[NODE:sufficiency_judge] chunks_count={len(chunks)}")

    # Call the judge (uses Gemini key 0 internally)
    result = await run_sufficiency_judge(
        question=last_message,
        retrieved_context=retrieved_context,
    )

    verdict = result.get("verdict", "PARTIAL")
    reason = result.get("reason", "")

    print(f"[NODE:sufficiency_judge] verdict={verdict}")
    print(f"[NODE:sufficiency_judge] reason='{reason}'")

    return {
        "judge_verdict": verdict,
        "judge_reason": reason,
    }


# ============================================================================
# AwaitFallbackNode — no-op interrupt point for PARTIAL/INSUFFICIENT
# ============================================================================

async def await_fallback_node(state: AgentState) -> dict:
    """
    AwaitFallbackNode: No-op node that acts as the LangGraph interrupt point.

    LangGraph fires 'interrupt_before' this node when the judge returns
    PARTIAL or INSUFFICIENT. Execution pauses here. The frontend receives a
    'fallback_choice_pending' SSE event and prompts the user to choose a strategy.

    When the user posts to POST /chat/{session_id}/choose-fallback, the graph
    resumes and route_after_fallback_choice reads fallback_strategy from state.

    This node does nothing — it only exists as an interrupt anchor.
    """
    print(f"[NODE:await_fallback] Waiting for user fallback choice... verdict={state.get('judge_verdict')}")
    # LangGraph requires nodes to write at least one field — pass judge_verdict through unchanged.
    return {"judge_verdict": state.get("judge_verdict")}


# ============================================================================
# TavilySearchNode — web search fallback
# ============================================================================

async def tavily_search_node(state: AgentState) -> dict:
    """
    TavilySearchNode: Perform a Tavily web search for the user's question.
    Stores results in web_results state field.

    Logs:
        [NODE:tavily_search] query
        [NODE:tavily_search] results_count
        [NODE:tavily_search] result titles
    """
    print("[NODE:tavily_search] Performing web search...")

    last_message = state["messages"][-1]["content"] if state["messages"] else ""
    print(f"[NODE:tavily_search] query='{last_message[:120]}'")

    results = await tavily_search(query=last_message, max_results=5)

    print(f"[NODE:tavily_search] results_count={len(results)}")
    for i, r in enumerate(results, 1):
        print(f"[NODE:tavily_search] result[{i}]: '{r.get('title', '')}' — {r.get('url', '')[:80]}")

    return {"web_results": results}


# ============================================================================
# PlannerNode — breaks complex requests into steps
# ============================================================================

PLANNER_SYSTEM_PROMPT = """You are a study session planner. Given a student's request, break it into an ordered list of concrete steps.

Your plan steps should be SHORT action descriptions (5-10 words each).
Return ONLY a numbered list of steps, nothing else.

For content_generation requests, typical steps:
1. Search notes for relevant content
2. Identify key concepts to cover
3. Generate the requested content type
4. Format the output

Return 3-6 steps maximum."""


async def planner_node(state: AgentState) -> dict:
    """
    PlannerNode: Decompose complex content_generation requests into an execution plan.
    Used ONLY for content_generation intent (study_planning now goes to gap_analysis).
    """
    print("[NODE:planner] Building execution plan...")

    llm = _get_groq_llm(temperature=0.2)
    last_message = state["messages"][-1]["content"] if state["messages"] else ""

    response = await llm.ainvoke([
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=f"Request: {last_message}"),
    ])

    # Parse numbered list into clean step strings
    raw_plan = response.content.strip()
    lines = raw_plan.split("\n")
    steps = []
    for line in lines:
        line = line.strip()
        if line and (line[0].isdigit() or line.startswith("-")):
            clean = line.lstrip("0123456789.-) ").strip()
            if clean:
                steps.append(clean)

    if not steps:
        steps = ["Search notes for relevant content", "Generate response"]

    print(f"[NODE:planner] Plan: {steps}")
    return {"plan": steps}


# ============================================================================
# GapAnalysisNode — knowledge gap analysis (Day 5)
# ============================================================================

async def gap_analysis_node(state: AgentState) -> dict:
    """
    GapAnalysisNode: Analyze which topics are covered/missing in the student's notes.

    Called as the first node for study_planning intent.
    Stores results in tool_results["gap_analysis"].
    """
    print("[NODE:gap_analysis] Running knowledge gap analysis...")

    session_id = state["session_id"]
    subject_name = state.get("subject_name") or "default"
    topics = state.get("topics", [])

    try:
        gap_result = knowledge_gap_analysis(
            session_id=session_id,
            subject_name=subject_name,
            custom_topics=topics,
        )
    except Exception as e:
        print(f"[NODE:gap_analysis] Error: {e}")
        gap_result = {"error": str(e), "well_covered": [], "shallow": [], "missing": []}

    tool_results = dict(state.get("tool_results") or {})
    tool_results["gap_analysis"] = gap_result

    print(
        f"[NODE:gap_analysis] covered={len(gap_result.get('well_covered', []))}, "
        f"shallow={len(gap_result.get('shallow', []))}, "
        f"missing={len(gap_result.get('missing', []))}"
    )
    return {"tool_results": tool_results}


# ============================================================================
# StudyPlanBuilderNode — generates proposed study plan (Day 5)
# ============================================================================

# Prompt to extract exam date + subject name from user message
EXTRACT_PLAN_PROMPT = """Extract study planning information from the student's message.
Return ONLY a JSON object with exactly these fields:
  - "exam_date": the exam date as a string (e.g. "April 15", "2025-04-15") or null if not mentioned
  - "subject": the subject/topic name or null if not mentioned

Examples:
  "plan my OS revision, exam is April 15" → {"exam_date": "April 15", "subject": "Operating Systems"}
  "I have a databases exam on June 3rd" → {"exam_date": "June 3", "subject": "databases"}
  "help me study" → {"exam_date": null, "subject": null}

Return ONLY the JSON, no markdown, no explanation."""


async def study_plan_builder_node(state: AgentState) -> dict:
    """
    StudyPlanBuilderNode: Generate a proposed study schedule.

    1. Extracts exam_date and subject_name from the user message
    2. Calls generate_study_plan to build weighted session list
    3. Stores proposed events in state
    4. Sets awaiting_confirmation = True (triggers interrupt before CalendarNode)
    """
    print("[NODE:study_plan_builder] Building study plan...")

    import json as _json
    llm = _get_groq_llm(temperature=0.0)
    last_message = state["messages"][-1]["content"] if state["messages"] else ""

    # Step 1: Extract exam date + subject from the message
    extract_response = await llm.ainvoke([
        SystemMessage(content=EXTRACT_PLAN_PROMPT),
        HumanMessage(content=last_message),
    ])

    raw = extract_response.content.strip()
    # Strip markdown code fences if present
    clean = re.sub(r"```(?:json)?\n?", "", raw).strip().rstrip("`")

    try:
        extracted = _json.loads(clean)
    except Exception:
        extracted = {"exam_date": None, "subject": None}

    exam_date = extracted.get("exam_date") or state.get("exam_date") or "2025-12-31"
    subject_name = (
        extracted.get("subject")
        or state.get("subject_name")
        or "General"
    )

    print(f"[NODE:study_plan_builder] exam_date='{exam_date}', subject='{subject_name}'")

    # Step 2: Generate study plan
    topics = state.get("topics", [])
    try:
        plan_result = generate_study_plan(
            session_id=state["session_id"],
            subject_name=subject_name,
            exam_date=exam_date,
            custom_topics=topics,
        )
    except Exception as e:
        print(f"[NODE:study_plan_builder] Error generating plan: {e}")
        tool_results = dict(state.get("tool_results") or {})
        tool_results["study_plan"] = {"error": str(e)}
        return {
            "proposed_calendar_events": [],
            "awaiting_confirmation": False,
            "tool_results": tool_results,
        }

    proposed_events = plan_result.get("proposed_events", [])

    # Update tool_results with full plan data
    tool_results = dict(state.get("tool_results") or {})
    tool_results["study_plan"] = plan_result

    print(f"[NODE:study_plan_builder] {len(proposed_events)} sessions proposed")
    return {
        "proposed_calendar_events": proposed_events,
        "awaiting_confirmation": True,
        "exam_date": exam_date,
        "subject_name": subject_name,
        "tool_results": tool_results,
    }


# ============================================================================
# CalendarNode — creates Google Calendar events (Day 5, interrupt fires before this)
# ============================================================================

async def calendar_node(state: AgentState) -> dict:
    """
    CalendarNode: Create Google Calendar events for each proposed study session.

    This node fires ONLY after the user confirms the plan via POST /confirm-plan.
    LangGraph interrupt fires BEFORE this node — execution resumes here on confirm.

    Calls create_study_session_event for each event in proposed_calendar_events.
    Stores created event links in tool_results["calendar_events"].
    """
    print("[NODE:calendar] Creating Google Calendar events...")

    from routes.deps import get_db
    db = get_db()

    user_id = state["user_id"]
    proposed_events = state.get("proposed_calendar_events", [])

    if not proposed_events:
        print("[NODE:calendar] No events to create")
        tool_results = dict(state.get("tool_results") or {})
        tool_results["calendar_events"] = []
        return {
            "tool_results": tool_results,
            "awaiting_confirmation": False,
        }

    created_events = []
    failed_events = []

    for event in proposed_events:
        try:
            result = await create_study_session_event(
                user_id=user_id,
                subject=event.get("subject", "Study"),
                date=event.get("date"),
                duration_minutes=event.get("duration_minutes", 60),
                db=db,
                start_time=event.get("start_time"),
                topic=event.get("topic"),
            )
        except Exception as e:
            print(f"[NODE:calendar] Exception creating event on {event.get('date')}: {e}")
            result = {"error": str(e)}

        if "error" in result:
            print(f"[NODE:calendar] Failed to create event on {event['date']}: {result['error']}")
            failed_events.append({**event, "error": result["error"]})
        else:
            created_events.append(result)
            print(f"[NODE:calendar] Created: {result.get('html_link', '')}")

    tool_results = dict(state.get("tool_results") or {})
    tool_results["calendar_events"] = created_events
    if failed_events:
        tool_results["calendar_failed"] = failed_events

    print(f"[NODE:calendar] Done: {len(created_events)} created, {len(failed_events)} failed")
    return {
        "tool_results": tool_results,
        "awaiting_confirmation": False,
    }


# ============================================================================
# DirectCalendarBuilderNode — direct calendar event creation (Day 5 feature)
# ============================================================================

EXTRACT_DIRECT_CALENDAR_PROMPT = """Extract calendar event parameters from the student's message.
Return ONLY a JSON object with exactly these fields:
  - "date": the date of the event strictly in YYYY-MM-DD format. Today's date is {today}.
  - "start_time": the start time strictly in HH:MM format (24-hour). If not specified, default to "09:00".
  - "topic": the subject/topic to study (e.g. "Operating Systems", "React Hooks")
  - "duration_minutes": integer duration in minutes (default to 60 if not specified)

Examples (assuming today is 2025-04-10):
  "schedule a 90 min study block for databases tomorrow at 3pm" → {"date": "2025-04-11", "start_time": "15:00", "topic": "databases", "duration_minutes": 90}
  "add a 2 hour math prep session on Friday" → {"date": "2025-04-14", "start_time": "09:00", "topic": "math prep", "duration_minutes": 120}

Return ONLY the JSON, no markdown, no explanation."""


async def direct_calendar_builder_node(state: AgentState) -> dict:
    """
    DirectCalendarBuilderNode: Schedule a calendar event directly from a user message.
    """
    print("[NODE:direct_calendar_builder] Building direct calendar event...")

    import json as _json
    llm = _get_groq_llm(temperature=0.0)
    last_message = state["messages"][-1]["content"] if state["messages"] else ""

    from datetime import datetime
    today_str = datetime.now().strftime("%Y-%m-%d")
    system_prompt = EXTRACT_DIRECT_CALENDAR_PROMPT.replace("{today}", today_str)

    # Step 1: Extract date, topic, duration
    extract_response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=last_message),
    ])

    raw = extract_response.content.strip()
    clean = re.sub(r"```(?:json)?\n?", "", raw).strip().rstrip("`")

    try:
        extracted = _json.loads(clean)
    except Exception:
        extracted = {"date": "tomorrow", "start_time": "09:00", "topic": "Study Session", "duration_minutes": 60}

    date_str = extracted.get("date", today_str)
    start_time_str = extracted.get("start_time", "09:00")
    topic = extracted.get("topic", state.get("subject_name", "General Study"))
    duration = int(extracted.get("duration_minutes", 60))

    # Format into proposed_calendar_events array
    event = {
        "date": date_str,
        "start_time": start_time_str,
        "subject": state.get("subject_name", "Study"),
        "topic": topic,
        "duration_minutes": duration,
        "reasoning": "Direct user request",
    }

    proposed_events = [event]

    tool_results = dict(state.get("tool_results") or {})
    tool_results["direct_calendar"] = {
        "status": "proposed",
        "event": event
    }

    print(f"[NODE:direct_calendar_builder] Event proposed: {event}")
    return {
        "proposed_calendar_events": proposed_events,
        "awaiting_confirmation": True,
        "tool_results": tool_results,
    }


# ============================================================================
# FlashcardGeneratorNode — generates and persists flashcards (Day 5)
# ============================================================================

EXTRACT_FLASHCARD_PROMPT = """Extract flashcard generation parameters from the student's message.
Return ONLY a JSON object with exactly these fields:
  - "topic": the topic to create flashcards on (string)
  - "num_cards": number of cards requested (integer, default 10 if not specified)

Examples:
  "make me 5 flashcards on scheduling algorithms" → {"topic": "scheduling algorithms", "num_cards": 5}
  "create flashcards for recursion" → {"topic": "recursion", "num_cards": 10}
  "give me 20 cards on the French Revolution" → {"topic": "French Revolution", "num_cards": 20}

Return ONLY the JSON, no markdown, no explanation."""


async def flashcard_generator_node(state: AgentState) -> dict:
    """
    FlashcardGeneratorNode: Generate flashcards using RAG + Gemini and persist to Atlas.

    Extracts topic and num_cards from the user message, then calls create_flashcards.
    Stores result in tool_results["flashcards"].
    """
    print("[NODE:flashcard_generator] Generating flashcards...")

    import json as _json
    from routes.deps import get_db
    db = get_db()

    llm = _get_groq_llm(temperature=0.2)
    last_message = state["messages"][-1]["content"] if state["messages"] else ""

    # Step 1: Extract topic + num_cards
    extract_response = await llm.ainvoke([
        SystemMessage(content=EXTRACT_FLASHCARD_PROMPT),
        HumanMessage(content=last_message),
    ])

    raw = extract_response.content.strip()
    clean = re.sub(r"```(?:json)?\n?", "", raw).strip().rstrip("`")

    try:
        extracted = _json.loads(clean)
        topic = extracted.get("topic", "General Study Topic")
        num_cards = int(extracted.get("num_cards", 10))
        num_cards = max(1, min(num_cards, 30))  # cap between 1–30
    except Exception:
        topic = last_message[:50]
        num_cards = 10

    print(f"[NODE:flashcard_generator] topic='{topic}', num_cards={num_cards}")

    # Step 2: Create flashcards
    try:
        flashcard_result = await create_flashcards(
            topic=topic,
            num_cards=num_cards,
            session_id=state["session_id"],
            subject_id=state["subject_id"],
            user_id=state["user_id"],
            db=db,
            llm=llm,
        )
    except Exception as e:
        print(f"[NODE:flashcard_generator] Error: {e}")
        flashcard_result = {"error": str(e), "cards_created": 0}

    tool_results = dict(state.get("tool_results") or {})
    tool_results["flashcards"] = flashcard_result

    print(f"[NODE:flashcard_generator] Created {flashcard_result.get('cards_created', 0)} flashcards")
    return {"tool_results": tool_results}


# ============================================================================
# RevisionSheetNode — generates exam revision document (Day 5)
# ============================================================================

async def revision_sheet_node(state: AgentState) -> dict:
    """
    RevisionSheetNode: Generate a structured exam revision sheet.

    Runs gap_analysis + summarize_topic for each topic, compiles markdown doc.
    Stores result in tool_results["revision_sheet"].
    """
    print("[NODE:revision_sheet] Generating revision sheet...")

    llm = _get_llm(temperature=0.3)

    # Cool down before starting — previous nodes (router, flashcard, synthesis)
    # may have consumed some budget, but since Llama handles router now, 2s is enough.
    print("[NODE:revision_sheet] Waiting 2s for API rate limit window reset...")
    import asyncio as _asyncio
    await _asyncio.sleep(2)

    subject_name = state.get("subject_name") or "General"
    topics = state.get("topics", [])

    # Step 2: Generate sheet (loops over topics + gap analysis)
    try:
        result = await generate_exam_revision_sheet(
            subject_id=state["subject_id"],
            session_id=state["session_id"],
            subject_name=subject_name,
            llm=llm,
            custom_topics=topics,
        )
    except Exception as e:
        print(f"[NODE:revision_sheet] Error: {e}")
        result = {"error": str(e)}

    tool_results = dict(state.get("tool_results") or {})
    tool_results["revision_sheet"] = result

    print(f"[NODE:revision_sheet] Done: {result.get('topics_covered', 0)} topics")
    return {"tool_results": tool_results}


# ============================================================================
# TranslationNode — extracts text/lang and translates (Day 5)
# ============================================================================

EXTRACT_TRANSLATION_PROMPT = """Extract translation parameters from the student's message.
Return ONLY a JSON object with exactly these fields:
  - "target_language": the language to translate to (e.g. "French", "Spanish")
  - "text": the exact text that needs to be translated

Examples:
  "translate this to French: The study agent is working perfectly!" → {"target_language": "French", "text": "The study agent is working perfectly!"}
  "how do you say hello world in spanish?" → {"target_language": "Spanish", "text": "hello world"}

Return ONLY the JSON, no markdown, no explanation."""

async def translation_node(state: AgentState) -> dict:
    """
    TranslationNode: Translate text using Hugging Face NLLB.
    Extracts text and target language, then calls translate_content.
    """
    print("[NODE:translation] Translating content...")

    import json as _json
    llm = _get_groq_llm(temperature=0.0)
    last_message = state["messages"][-1]["content"] if state["messages"] else ""

    # Step 1: Extract text + language
    extract_response = await llm.ainvoke([
        SystemMessage(content=EXTRACT_TRANSLATION_PROMPT),
        HumanMessage(content=last_message),
    ])

    raw = extract_response.content.strip()
    clean = re.sub(r"```(?:json)?\n?", "", raw).strip().rstrip("`")

    try:
        extracted = _json.loads(clean)
        target_language = extracted.get("target_language", "English")
        text_to_translate = extracted.get("text", last_message)
    except Exception:
        target_language = "English"
        text_to_translate = last_message

    print(f"[NODE:translation] target_language='{target_language}'")

    # Step 2: Translate
    try:
        translation_result = translate_content(text=text_to_translate, target_language=target_language)
    except Exception as e:
        print(f"[NODE:translation] Error: {e}")
        translation_result = {"error": str(e)}

    tool_results = dict(state.get("tool_results") or {})
    tool_results["translation"] = translation_result

    return {"tool_results": tool_results}


# ============================================================================
# SynthesisNode — the ONLY node that produces the final answer
# Five distinct generation paths keyed by (judge_verdict, fallback_strategy)
# ============================================================================

# Base system prompt for the answer-generation Gemini instance
SYNTHESIS_SYSTEM_PROMPT = """You are a knowledgeable, encouraging study assistant named SKB (Study Knowledge Bot).

Your job is to give clear, accurate, and helpful responses to students.

RULES:
- Always prioritize information found in the student's uploaded notes
- Be conversational but educational
- Use bullet points for lists, but DO NOT use asterisks (**) for bolding. Use plain text instead.
- Keep responses focused and appropriately detailed for studying
- Never make up facts — if uncertain, say so
- When external information is used, label it clearly and separately from note content
- Never silently mix note content and external content
- Automatically detect the language the user is speaking, and seamlessly reply in that same language."""


# ---- Case prompts ----

def _build_sufficient_prompt(question: str, retrieved_context: str) -> str:
    """CASE 1 — SUFFICIENT: Answer from notes only."""
    return f"""You are a study assistant.

Answer the question using ONLY the provided notes.

If information is missing from the notes, explicitly state that rather than guessing.

Question:
{question}

Notes:
{retrieved_context}"""


def _build_partial_gemini_prompt(question: str, retrieved_context: str) -> str:
    """CASE 2 — PARTIAL + Gemini: Notes first, then supplement with Gemini knowledge."""
    return f"""The notes partially answer the question.

Use the notes first. Then provide additional explanation using your own knowledge.

Format your response EXACTLY as:

## From the Notes

(content from notes)

## Additional Explanation

(content from your own knowledge, clearly separated)

Question:
{question}

Notes:
{retrieved_context}"""


def _build_partial_tavily_prompt(question: str, retrieved_context: str, web_results_text: str) -> str:
    """CASE 3 — PARTIAL + Tavily: Notes first, then web results to fill gaps."""
    return f"""The notes partially answer the question. Web search results are provided to fill the gaps.

Instructions:
1. Prioritize information found in the notes.
2. Use web results only to fill missing gaps.
3. Clearly separate note content and web content.

Format your response EXACTLY as:

## From the Notes

(content from notes)

## Additional Information from Web Search

(content from web search results, with sources cited)

Question:
{question}

Notes:
{retrieved_context}

Web Results:
{web_results_text}"""


def _build_insufficient_gemini_prompt(question: str) -> str:
    """CASE 4 — INSUFFICIENT + Gemini: Answer using Gemini knowledge, with warning."""
    return f"""The uploaded notes do not adequately cover this topic.

Answer using your own knowledge.

IMPORTANT: Begin your answer with EXACTLY this line:
"⚠ This topic is not adequately covered in the uploaded notes."

Then provide a complete, helpful answer.

Question:
{question}"""


def _build_insufficient_tavily_prompt(question: str, web_results_text: str) -> str:
    """CASE 5 — INSUFFICIENT + Tavily: Answer from web results, with warning."""
    return f"""The uploaded notes do not adequately cover this topic.

Generate a complete answer using the web results below.

IMPORTANT: Begin your answer with EXACTLY this line:
"⚠ This topic is not adequately covered in the uploaded notes."

Then provide a complete answer based on the web results, citing sources where appropriate.

Question:
{question}

Web Results:
{web_results_text}"""


async def synthesis_node(state: AgentState, config: RunnableConfig) -> dict:
    """
    SynthesisNode: Generate the final response using all available context.

    Routes to one of 5 answer generation cases based on (judge_verdict, fallback_strategy):
      SUFFICIENT                  → Case 1: notes only
      PARTIAL   + gemini strategy → Case 2: notes + Gemini knowledge
      PARTIAL   + tavily strategy → Case 3: notes + web results
      INSUFFICIENT + gemini       → Case 4: Gemini knowledge + warning
      INSUFFICIENT + tavily       → Case 5: web results + warning

    Uses Gemini key 1 (answer key) exclusively via _get_answer_llm().
    Also handles all non-RAG paths (flashcards, study plans, calendar, chitchat).
    """
    print("[NODE:synthesis] Generating final response...")

    # Use the dedicated answer LLM (Gemini key 1)
    llm = _get_answer_llm(temperature=0.4)

    last_message = state["messages"][-1]["content"] if state["messages"] else ""
    chunks = state.get("retrieved_chunks") or []
    tool_results = state.get("tool_results") or {}
    intent = state.get("intent", "rag_query")
    plan = state.get("plan") or []

    # ---- Judge state ----
    judge_verdict = state.get("judge_verdict")
    fallback_strategy = state.get("fallback_strategy")
    web_results = state.get("web_results") or []

    # ---- Build formatted retrieved context (shared across RAG cases) ----
    if chunks:
        retrieved_context = "\n\n---\n\n".join([
            f"[From: {c['metadata'].get('filename', 'notes')}]\n{c['content']}"
            for c in chunks
        ])
    else:
        retrieved_context = "No relevant content was retrieved from the notes."

    # ---- Determine answer path and final answer_source ----
    answer_source = "notes"  # default
    user_prompt = None

    # RAG query path — route by judge verdict
    if intent == "rag_query" and judge_verdict:
        if judge_verdict == "SUFFICIENT":
            # Case 1 — notes only
            user_prompt = _build_sufficient_prompt(last_message, retrieved_context)
            answer_source = "notes"
            print("[SYNTHESIS] case=1 (SUFFICIENT, notes only)")

        elif judge_verdict == "PARTIAL":
            if fallback_strategy == "tavily" and web_results:
                # Case 3 — notes + Tavily
                web_results_text = format_web_results_for_prompt(web_results)
                user_prompt = _build_partial_tavily_prompt(last_message, retrieved_context, web_results_text)
                answer_source = "notes+tavily"
                print(f"[SYNTHESIS] case=3 (PARTIAL + tavily), web_results_count={len(web_results)}")
            else:
                # Case 2 — notes + Gemini (default for PARTIAL with gemini or no strategy)
                user_prompt = _build_partial_gemini_prompt(last_message, retrieved_context)
                answer_source = "notes+gemini"
                print("[SYNTHESIS] case=2 (PARTIAL + gemini)")

        elif judge_verdict == "INSUFFICIENT":
            if fallback_strategy == "tavily" and web_results:
                # Case 5 — Tavily only
                web_results_text = format_web_results_for_prompt(web_results)
                user_prompt = _build_insufficient_tavily_prompt(last_message, web_results_text)
                answer_source = "tavily"
                print(f"[SYNTHESIS] case=5 (INSUFFICIENT + tavily), web_results_count={len(web_results)}")
            else:
                # Case 4 — Gemini only
                user_prompt = _build_insufficient_gemini_prompt(last_message)
                answer_source = "gemini"
                print("[SYNTHESIS] case=4 (INSUFFICIENT + gemini)")

    # ---- Non-RAG paths (Day 5 features, chitchat, etc.) ----

    # Build context for non-RAG paths
    context_parts = []

    if chunks and intent != "rag_query":
        notes_text = "\n\n---\n\n".join([
            f"[From: {c['metadata'].get('filename', 'notes')}]\n{c['content']}"
            for c in chunks
        ])
        confidence = state.get("chunk_confidence", 0.0)
        context_parts.append(f"## Notes from Student's Documents (confidence: {confidence:.0%})\n\n{notes_text}")

    gap = tool_results.get("gap_analysis", {})
    if gap:
        context_parts.append(
            f"## Knowledge Gap Analysis\n"
            f"- Well covered: {', '.join(gap.get('well_covered', [])) or 'none'}\n"
            f"- Needs work: {', '.join(gap.get('shallow', [])) or 'none'}\n"
            f"- Missing: {', '.join(gap.get('missing', [])) or 'none'}"
        )

    # Day 5: Study plan context (awaiting confirmation)
    study_plan = tool_results.get("study_plan", {})
    if study_plan and state.get("awaiting_confirmation"):
        proposed = study_plan.get("proposed_events", [])
        exam_date = study_plan.get("exam_date", "")
        sessions_summary = "\n".join([
            f"  • {e['date']}: {e['topic']} ({e['duration_minutes']} min)"
            for e in proposed[:10]  # show first 10
        ])
        context_parts.append(
            f"## Proposed Study Plan ({len(proposed)} sessions until {exam_date})\n{sessions_summary}"
        )

    # Direct calendar (awaiting confirmation)
    direct_calendar = tool_results.get("direct_calendar", {})
    if direct_calendar and state.get("awaiting_confirmation"):
        event = direct_calendar.get("event", {})
        context_parts.append(
            f"## Proposed Calendar Event\n  • {event.get('date')}: {event.get('topic')} ({event.get('duration_minutes')} min)"
        )

    # Day 5: Calendar events just created
    calendar_events = tool_results.get("calendar_events", [])
    if calendar_events:
        links = "\n".join([
            f"  • {e.get('title', '')}: {e.get('html_link', 'no link')}"
            for e in calendar_events
        ])
        context_parts.append(f"## Google Calendar Events Created\n{links}")

    # Day 5: Flashcards just generated
    flashcard_data = tool_results.get("flashcards", {})
    if flashcard_data and flashcard_data.get("cards_created", 0) > 0:
        cards = flashcard_data.get("cards", [])
        cards_text = "\n".join([
            f"  **Q:** {c['question']}\n  **A:** {c['answer']}"
            for c in cards[:5]  # preview first 5
        ])
        context_parts.append(
            f"## Flashcards Generated ({flashcard_data['cards_created']} total)\n{cards_text}"
        )

    # Day 5: Revision sheet
    revision_sheet = tool_results.get("revision_sheet", {})
    if revision_sheet and revision_sheet.get("document"):
        doc_preview = revision_sheet["document"][:1500]  # first 1500 chars
        context_parts.append(f"## Revision Sheet (Preview)\n{doc_preview}")

    # Day 5: Translation
    translation_result = tool_results.get("translation", {})
    if translation_result and "translated_text" in translation_result:
        context_parts.append(f"## Translation Result\n{translation_result['translated_text']}")

    context_block = "\n\n".join(context_parts) if context_parts else "No specific context available — using general knowledge."

    # ---- Build intent-aware prompt for non-RAG paths ----
    if user_prompt is None:
        if intent == "chitchat":
            user_prompt = f"Student said: {last_message}\n\nRespond naturally and warmly."
            answer_source = "gemini"

        elif state.get("awaiting_confirmation") and study_plan:
            proposed = study_plan.get("proposed_events", [])
            exam_date = study_plan.get("exam_date", "")
            user_prompt = f"""Student request: {last_message}

I've analyzed their knowledge gaps and created a study plan.

{context_block}

Present the proposed study plan clearly:
- Show the key sessions (group by week if many)
- Explain WHY certain topics are prioritized (based on gap analysis)
- End with: "Reply confirm to add these to your Google Calendar, or let me know if you'd like to adjust the plan."

Keep the tone encouraging and motivating."""
            answer_source = "notes"

        elif state.get("awaiting_confirmation") and tool_results.get("direct_calendar"):
            user_prompt = f"""Student request: {last_message}

I've extracted the details for a calendar event based on their request.

{context_block}

Present this proposed event clearly and end with: "Reply confirm to add this to your Google Calendar, or let me know if you'd like to adjust it."
Keep the tone encouraging and helpful."""
            answer_source = "notes"

        elif flashcard_data and flashcard_data.get("cards_created", 0) > 0:
            user_prompt = f"""Student request: {last_message}

{context_block}

Present the flashcards in a clear, readable format. Show ALL {flashcard_data.get('cards_created', 0)} cards.
Format your output EXACTLY like this:
Card N: [card_type]
Q: [question]
A: [answer]

End with an encouraging message about using the cards for spaced repetition."""
            answer_source = "notes"

        elif calendar_events:
            user_prompt = f"""The study sessions have been added to Google Calendar!

{context_block}

Summarize what was created, list the calendar links, and give an encouraging message about following the study schedule."""
            answer_source = "notes"

        elif revision_sheet and revision_sheet.get("document"):
            user_prompt = f"Return the following revision document verbatim:\n\n{revision_sheet['document']}"
            answer_source = "notes"

        elif translation_result and "translated_text" in translation_result:
            user_prompt = f"Return the following translated text clearly to the user:\n\n{translation_result['translated_text']}"
            answer_source = "notes"

        elif plan:
            plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
            user_prompt = f"""Student request: {last_message}

Execution plan:
{plan_str}

Context:
{context_block}

Execute the plan and fulfill the student's request completely."""
            answer_source = "notes"

        else:
            # Generic fallback
            user_prompt = f"""Student question: {last_message}

Context from notes and tools:
{context_block}

Answer the question using the context above."""
            answer_source = "notes"

    # ---- Call Gemini (answer key) ----
    response = await llm.ainvoke([
        SystemMessage(content=SYNTHESIS_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ], config=config)

    answer = response.content if hasattr(response, "content") else str(response)

    print(f"[SYNTHESIS] answer_source={answer_source}")
    print(f"[SYNTHESIS] fallback_strategy={fallback_strategy}")
    print(f"[SYNTHESIS] web_results_count={len(web_results)}")
    print(f"[NODE:synthesis] Generated response ({len(answer)} chars)")

    return {
        "response": answer,
        "answer_source": answer_source,
    }


# ============================================================================
# Edge routing functions (used by LangGraph conditional edges)
# ============================================================================

def route_by_intent(state: AgentState) -> Literal["rag", "planner", "gap_analysis", "direct_calendar_builder", "synthesis", "translation", "evaluator"]:
    """
    Map intent → next node after router.
    """
    routes = {
        "rag_query": "rag",
        "content_generation": "planner",
        "study_planning": "gap_analysis",
        "calendar_scheduling": "direct_calendar_builder",
        "session_end": "evaluator",
        "chitchat": "synthesis",
        "translation": "translation",
    }
    intent = state.get("intent", "rag_query")
    print(f"[EDGE:intent] {intent} -> {routes.get(intent, 'rag')}")
    return routes.get(intent, "rag")


def route_after_judge(state: AgentState) -> Literal["synthesis", "await_fallback"]:
    """
    After SufficiencyJudgeNode:
    - SUFFICIENT → synthesis (no user intervention needed)
    - PARTIAL / INSUFFICIENT → await_fallback (interrupt fires, user chooses strategy)
    """
    verdict = state.get("judge_verdict", "INSUFFICIENT")
    if verdict == "SUFFICIENT":
        print(f"[EDGE:judge] verdict=SUFFICIENT → synthesis")
        return "synthesis"
    print(f"[EDGE:judge] verdict={verdict} → await_fallback (interrupt)")
    return "await_fallback"


def route_after_fallback_choice(state: AgentState) -> Literal["tavily_search", "synthesis"]:
    """
    After AwaitFallbackNode (user has chosen their strategy):
    - tavily → tavily_search_node then synthesis
    - gemini  → synthesis directly
    """
    strategy = state.get("fallback_strategy", "gemini")
    if strategy == "tavily":
        print(f"[EDGE:fallback_choice] strategy=tavily → tavily_search")
        return "tavily_search"
    print(f"[EDGE:fallback_choice] strategy=gemini → synthesis")
    return "synthesis"


def route_by_content_type(state: AgentState) -> Literal["flashcard_generator", "revision_sheet", "rag"]:
    """
    After PlannerNode (content_generation intent), decide which specialist node to use.

    Detects flashcard and revision sheet requests from the last message.
    Falls back to rag (which then goes to synthesis) for everything else.
    """
    last_message = (state["messages"][-1]["content"] if state["messages"] else "").lower()

    flashcard_keywords = {"flashcard", "flash card", "cards", "quiz cards", "anki"}
    revision_keywords = {"revision sheet", "revision document", "cheat sheet", "study sheet", "summary sheet"}

    if any(kw in last_message for kw in flashcard_keywords):
        print("[EDGE:content_type] → flashcard_generator")
        return "flashcard_generator"

    if any(kw in last_message for kw in revision_keywords):
        print("[EDGE:content_type] → revision_sheet")
        return "revision_sheet"

    print("[EDGE:content_type] → rag (generic content generation)")
    return "rag"


async def route_by_calendar_auth(state: AgentState) -> Literal["has_auth", "no_auth"]:
    """
    Check if the user has Google OAuth tokens linked in Atlas.
    Returns 'has_auth' if tokens exist, 'no_auth' otherwise.
    """
    from routes.deps import get_db
    from bson import ObjectId

    user_id = state["user_id"]
    try:
        db = get_db()
        user = await db.users.find_one({"_id": ObjectId(user_id)})
        if user and user.get("google_access_token"):
            print(f"[EDGE:calendar_auth] user {user_id} → has_auth")
            return "has_auth"
    except Exception as e:
        print(f"[EDGE:calendar_auth] Error checking auth: {e}")

    print(f"[EDGE:calendar_auth] user {user_id} → no_auth")
    return "no_auth"


# ============================================================================
# EvaluatorNode — assesses the quality of the completed session (Day 7)
# ============================================================================

EVALUATOR_SYSTEM_PROMPT = """You are an expert educational evaluator. Analyze the provided study session transcript and return a structured JSON assessment.

Evaluate:
1. topics_covered: list of distinct topics/concepts discussed
2. depth: for each topic, assign one of "surface" | "moderate" | "deep" based on how thoroughly it was explored
3. weak_moments: list of moments where the student showed confusion, incorrect understanding, or needed significant clarification (brief descriptions)
4. session_score: an integer 1-10 rating of the session quality (10=excellent engagement and understanding)
5. strong_areas: topics the student clearly understood well

Return ONLY valid JSON in this exact format:
{
  "topics_covered": ["topic1", "topic2"],
  "depth": {"topic1": "deep", "topic2": "surface"},
  "weak_moments": ["student confused X with Y", "struggled to explain Z"],
  "strong_areas": ["topic1"],
  "session_score": 7
}

If the transcript has no real study content (e.g., just greetings), return a score of 1 and empty lists."""


async def evaluator_node(state: AgentState) -> dict:
    """
    EvaluatorNode: Assess session quality by reading the full transcript.
    Fires when intent == session_end.
    Stores structured evaluation in state["evaluation"].
    """
    import json as _json
    print("[NODE:evaluator] Running session evaluation...")

    llm = _get_judge_llm(temperature=0.1)

    transcript = state.get("transcript", []) or state.get("messages", [])

    if not transcript:
        print("[NODE:evaluator] No transcript — skipping evaluation")
        return {
            "evaluation": {
                "topics_covered": [],
                "depth": {},
                "weak_moments": [],
                "strong_areas": [],
                "session_score": 1,
            }
        }

    # Format transcript as readable text
    transcript_text = "\n".join([
        f"{msg.get('role', 'unknown').upper()}: {msg.get('content', '')[:500]}"
        for msg in transcript
        if msg.get("content") and msg.get("role") in ("user", "assistant")
    ])

    response = await llm.ainvoke([
        SystemMessage(content=EVALUATOR_SYSTEM_PROMPT),
        HumanMessage(content=f"Session transcript:\n\n{transcript_text}"),
    ])

    raw = response.content.strip()
    # Strip markdown fences if present
    clean = re.sub(r"```(?:json)?\n?", "", raw).strip().rstrip("`")

    try:
        evaluation = _json.loads(clean)
    except Exception:
        print(f"[NODE:evaluator] JSON parse failed, using fallback")
        evaluation = {
            "topics_covered": [],
            "depth": {},
            "weak_moments": [],
            "strong_areas": [],
            "session_score": 5,
        }

    score = evaluation.get("session_score", 5)
    topics = evaluation.get("topics_covered", [])
    print(f"[NODE:evaluator] score={score}, topics={topics}")
    return {"evaluation": evaluation}


# ============================================================================
# SummaryNode — generates a human-readable session recap (Day 7)
# ============================================================================

SUMMARY_SYSTEM_PROMPT = """You are a study coach writing a post-session summary for a student.

Based on the evaluation data provided, write a concise, encouraging session summary.

Include:
- A brief overview of what was covered
- 1-2 sentences on strong areas
- 1-2 sentences on areas needing more work
- A specific suggestion for next session

Keep it to 3-4 short paragraphs. Tone: warm, constructive, and motivating. Do NOT use bullet points or headers."""


async def summary_node(state: AgentState) -> dict:
    """
    SummaryNode: Takes evaluator output and writes a human-readable summary.
    Stores result in state["session_summary"].
    """
    print("[NODE:summary] Generating session summary...")

    llm = _get_judge_llm(temperature=0.5)

    evaluation = state.get("evaluation") or {}
    topics = evaluation.get("topics_covered", [])
    depth = evaluation.get("depth", {})
    weak_moments = evaluation.get("weak_moments", [])
    strong_areas = evaluation.get("strong_areas", [])
    score = evaluation.get("session_score", 5)

    eval_summary = f"""Session score: {score}/10
Topics covered: {', '.join(topics) or 'General study'}
Strong areas: {', '.join(strong_areas) or 'None identified'}
Areas needing work: {', '.join(weak_moments) or 'None identified'}
Depth breakdown: {', '.join([f"{t}: {d}" for t, d in depth.items()]) or 'N/A'}"""

    response = await llm.ainvoke([
        SystemMessage(content=SUMMARY_SYSTEM_PROMPT),
        HumanMessage(content=eval_summary),
    ])

    summary = response.content.strip() if hasattr(response, "content") else str(response)
    print(f"[NODE:summary] Summary generated ({len(summary)} chars)")
    return {"session_summary": summary}


# ============================================================================
# SaveNode — persists evaluation + summary to Atlas and cleans up Chroma (Day 7)
# ============================================================================

async def save_node(state: AgentState) -> dict:
    """
    SaveNode: Writes the session evaluation and summary to Atlas.
    Then deletes the ChromaDB collection.
    This is the final node in the session_end pipeline.
    """
    import asyncio as _asyncio
    from bson import ObjectId
    from datetime import datetime
    from db.chroma import delete_session_collection
    from routes.deps import get_db

    print("[NODE:save] Persisting session evaluation and summary...")

    session_id = state["session_id"]
    evaluation = state.get("evaluation") or {}
    session_summary = state.get("session_summary") or ""

    try:
        db = get_db()
        session_oid = ObjectId(session_id)
        ended_at = datetime.utcnow()

        await db.sessions.update_one(
            {"_id": session_oid},
            {
                "$set": {
                    "status": "completed",
                    "ended_at": ended_at,
                    "summary": session_summary,
                    "evaluator_scores": {
                        "topics_covered": evaluation.get("topics_covered", []),
                        "depth": evaluation.get("depth", {}),
                        "weak_moments": evaluation.get("weak_moments", []),
                        "strong_areas": evaluation.get("strong_areas", []),
                        "session_score": evaluation.get("session_score", 0),
                    },
                }
            },
        )
        print(f"[NODE:save] Session {session_id} marked completed in Atlas")
    except Exception as e:
        print(f"[NODE:save] Atlas update error: {e}")

    # Delete ChromaDB collection
    try:
        await _asyncio.to_thread(delete_session_collection, session_id)
        print(f"[NODE:save] ChromaDB collection deleted for session {session_id}")
    except Exception as e:
        print(f"[NODE:save] ChromaDB deletion error (non-fatal): {e}")

    return {"response": state.get("session_summary") or "Session completed successfully."}
