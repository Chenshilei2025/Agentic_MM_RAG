"""REFLECT skill — cross-modal conflict detection and gap analysis.

Core Design for EMNLP paper:
  - Cross-modal evidence alignment at decision layer
  - Detect conflicts when same aspect has incompatible findings across modalities
  - Identify gaps in important subtasks (importance >= 0.7 with low coverage)
  - Guide next action: ANSWER, ESCALATE (Tier-2 disclosure), REPLAN, or WAIT

This skill produces ADVICE for the Decision Agent (Claude Opus 4.7). The DA
uses this as one input alongside other signals before committing to a command.
If reflect fails, the main loop degrades to rule-based heuristics.
"""
from typing import Any, Dict, List, Optional


REFLECT_SKILL_SYSTEM = """\
You are the REFLECT skill of a multimodal RAG Decision Agent. Your job is to
diagnose the current evidence state and recommend the next action.

INPUTS you will see:
  - QUERY           the user's original question
  - SUBTASKS        your earlier decomposition (aspect, modality, importance)
  - STATUS_BOARD    one row per sub-agent with finding + coverage + confidence
  - AGREEMENTS      cross-modal agreement state per aspect

=============================================================================
CROSS-MODAL CONFLICT DETECTION
=============================================================================

A CONFLICT exists when:
  1. Same aspect has findings from >=2 modalities
  2. Findings are incompatible or contradictory
  3. Use AGREEMENTS[state="disagree"] as a STRONG signal

Example conflicts:
  - aspect="entity", doc_text says "X acquired Y" but video_text says "Y independent"
  - aspect="temporal", doc_visual shows "2023 timeline" but video_text says "2022"

For EACH conflict, specify:
  - aspect: the aspect tag where conflict occurs
  - agent_ids: which sub-agents disagree
  - reason: brief description of the contradiction

=============================================================================
GAP ANALYSIS
=============================================================================

A GAP exists when:
  1. Subtask importance >= 0.7 (ESSENTIAL or PRIMARY)
  2. Coverage < 0.5 (poorly covered) OR no sub-agent assigned
  3. Confidence across modalities is low (< 0.6)

For EACH gap, specify:
  - aspect: the aspect tag with insufficient coverage
  - modality: which modality is missing or weak
  - reason: why this is a gap (low coverage / no agent / low confidence)

=============================================================================
CAN_ANSWER DECISION
=============================================================================

can_answer = true ONLY if ALL conditions are met:
  1. NO disagree-state aspects in AGREEMENTS
  2. EVERY importance>=0.7 subtask has coverage>=0.7
  3. NO unresolved gaps
  4. Max confidence across all agents >= 0.8

Otherwise can_answer = false

=============================================================================
RECOMMENDED ACTION
=============================================================================

Choose ONE based on your diagnosis:

  "ANSWER"   — can_answer is true; synthesize and STOP
  "ESCALATE" — conflicts exist OR gaps but sketch available; request
                Tier-2 curated evidence from specific agents
  "REPLAN"   — important subtask wholly missing OR consistently failing;
                decomposition needs revision
  "WAIT"     — sub-agents still running OR insufficient signal

ESCALATION TARGETS:
  - For conflicts: list ALL disagreeing agents
  - For gaps: list agents with low coverage but potentially useful evidence

=============================================================================
OUTPUT FORMAT
=============================================================================

Strict JSON object, no prose, no code fences:
{
  "can_answer": <bool>,
  "conflicts": [
    {"aspect": "<str>", "agent_ids": ["<id1>", "<id2>"], "reason": "<str>"}
  ],
  "gaps": [
    {"aspect": "<str>", "modality": "<str>", "reason": "<str>"}
  ],
  "recommended_action": "ANSWER" | "ESCALATE" | "REPLAN" | "WAIT",
  "escalation_targets": ["<agent_id>", ...]
}
"""


def run(query: str, status_board: List[Dict[str, Any]],
        aspect_agreements: List[Dict[str, Any]],
        subtasks: List[Any],
        provider: Any) -> Optional[Dict[str, Any]]:
    """Execute the reflect skill. Returns a verdict dict, or None on any
    failure. Main loop degrades to rule-based heuristics on None.

    Args:
        query: User's original question
        status_board: Current agent snapshots (one per sub-agent)
        aspect_agreements: Cross-modal agreement state per aspect
        subtasks: Decomposed subtasks with aspect/modality/importance
        provider: LLM provider with .complete() method

    Returns:
        Verdict dict with keys: can_answer, conflicts, gaps,
        recommended_action, escalation_targets. None on failure.
    """
    if provider is None or not hasattr(provider, "complete"):
        return None
    provider_type = type(provider).__name__
    if provider_type in ("ScriptedProvider", "MockProvider"):
        return None

    user_msg = _build_user_prompt(query, status_board, aspect_agreements,
                                  subtasks)
    try:
        raw = provider.complete(
            messages=[{"role": "user", "content": user_msg}],
            system=REFLECT_SKILL_SYSTEM,
            max_tokens=1000)  # Increased for detailed conflict analysis
    except Exception:
        return None

    return _extract_json_object(raw)


def _build_user_prompt(query, board, agreements, subtasks) -> str:
    """Build a structured user prompt for the reflect skill.

    Highlights conflicts and gaps explicitly for the LLM to analyze.
    """
    import json

    # Serialize subtasks with emphasis on importance
    st_repr = []
    for s in (subtasks or []):
        st_repr.append({
            "id": s.id,
            "aspect": s.aspect,
            "modalities": s.modalities,
            "importance": s.importance,
            "description": s.description,
            "is_essential": s.importance >= 0.7  # Flag for gap analysis
        })

    # Build the payload with highlighted conflict/gap signals
    payload = {
        "query": query,
        "subtasks": st_repr,
        "status_board": board,
        "aspect_agreements": agreements,
        # Explicit signals to guide LLM attention
        "highlighted_conflicts": [
            a for a in (agreements or [])
            if a.get("agreement_state") == "disagree"
        ],
        "essential_subtasks": [
            s.id for s in (subtasks or [])
            if s.importance >= 0.7
        ]
    }

    return json.dumps(payload, default=str, ensure_ascii=False, indent=2)


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    import json
    import re
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", t).strip()
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None
