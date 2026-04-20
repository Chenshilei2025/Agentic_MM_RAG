"""DECOMPOSE skill — query → structured subtasks with fine-grained aspects.

Core Design: Multi-modal RAG with modality-isolated retrieval.
- Decision Agent (Claude Opus 4.7) decomposes query into subtasks
- Each subtask focuses on ONE modality to avoid noise interference
- Aspect taxonomy enables cross-modal evidence alignment at decision layer

Called once at run start. A good decomposition dictates the entire run's
trajectory — worth spending tokens here.
"""
from typing import Any, List, Optional


# ==============================================================================
# ASPECT TAXONOMY — fine-grained semantic categories for cross-modal alignment
# ==============================================================================
_ASPECT_TAXONOMY = """
ASPECT TAXONOMY — pick ONE per subtask from these 6 categories:

  event         — What happened? Occurrences, actions, incidents, episodes
                  Examples: "product launch in 2023", "system failure sequence",
                  "user authentication flow", "meeting outcomes"

  spatial       — Where? Locations, layouts, configurations, distributions
                  Examples: "office floor plan", "network topology",
                  "UI component positions", "geographic distribution"

  entity        — What/Who? Objects, people, organizations, concepts, entities
                  Examples: "company acquisition targets", "API parameters",
                  "team members involved", "system components"

  causal        — Why? Causes, effects, mechanisms, dependencies, relationships
                  Examples: "root cause of delay", "factors affecting accuracy",
                  "prerequisites for feature X", "impact of change Y"

  temporal      — When? Timeline, sequence, duration, frequency, chronology
                  Examples: "project milestones", "historical evolution",
                  "processing order", "time-based patterns"

  process       — How? Methods, procedures, workflows, algorithms, instructions
                  Examples: "deployment steps", "calculation method",
                  "authentication process", "data pipeline"

RULES:
  - One aspect per subtask. If query spans multiple aspects, create multiple subtasks.
  - Aspect choice must preserve semantic meaning for downstream embedding.
  - Cross-modal subtasks on the SAME aspect enable evidence alignment at decision layer.
"""


# ==============================================================================
# MODALITY SEMANTICS — modality-isolated retrieval design
# ==============================================================================
_MODALITY_SEMANTICS = """
MODALITY SEMANTICS — each subtask picks ONE modality to avoid noise:

  doc_text      — Textual content from documents: paragraphs, definitions,
                  quotes, structured data. Use for: claims, facts, entities,
                  formal statements, citations.

  doc_visual    — Visual content from documents: diagrams, charts, tables,
                  scanned figures. Use for: architecture diagrams, flowcharts,
                  data visualizations, screenshots.

  video_text    — Transcribed speech from videos: narration, dialogue, ASR with
                  timestamps. Use for: what someone SAID, lectures, interviews,
                  spoken explanations.

  video_visual  — Visual frames from videos: on-screen objects, actions, scenes.
                  Use for: demonstrations, physical actions, UI interactions.

CROSS-MODAL CORROBORATION:
  - For verification tasks, create SAME-aspect subtasks in different modalities.
  - Example: aspect="entity", modality="doc_text" AND aspect="entity", modality="video_text"
  - Decision Agent will later align and compare findings across modalities.
"""


# ==============================================================================
# IMPORTANCE CALIBRATION
# ==============================================================================
_IMPORTANCE_CALIBRATION = """
IMPORTANCE CALIBRATION — anchor your numbers:

  0.9-1.0   ESSENTIAL — Query is UNANSWERABLE without this aspect.
  0.7-0.9   PRIMARY   — Central to answer, user explicitly expects this.
  0.5-0.7   SUPPORTING — Adds useful context, not strictly required.
  0.3-0.5   NICE_TO_HAVE — Background info, could be omitted in brief answer.
  <0.3      DO NOT DECOMPOSE — Merge into a sibling subtask.

CONSTRAINT: At least ONE subtask must have importance >= 0.7.
"""


_DECOMPOSE_SKILL_SYSTEM = _ASPECT_TAXONOMY + "\n" + _MODALITY_SEMANTICS + "\n" + _IMPORTANCE_CALIBRATION + """

=============================================================================
ROLE: You are the DECOMPOSE skill of a multimodal RAG Decision Agent.

Your single job: turn the USER'S QUERY into a structured set of 1-5 subtasks
that enable modality-isolated retrieval followed by cross-modal alignment.

HARD CONSTRAINTS — VIOLATIONS WILL BE REJECTED:
  1. Each subtask picks ONE OR MORE modalities from {doc_text, doc_visual,
     video_text, video_visual}.
  2. (aspect, modality) pairs across subtasks MUST be unique.
  3. Prefer FEWER, higher-importance subtasks. Don't over-decompose.
  4. At least ONE subtask must have importance >= 0.7.
  5. Description MUST preserve semantic meaning — it will be embedded for
     similarity matching. Be specific but concise.

CROSS-MODAL SUBTASKS:
  - If a subtask needs multiple modalities, list them all in "modalities".
    Example: "modalities": ["doc_text", "doc_visual"] for text+diagram.
  - The orchestrator will split multi-modality subtasks into single-modality
    siblings automatically, preserving the same aspect tag.

OUTPUT FORMAT — strict JSON array, no prose, no code fences:
[
  {
    "id": "s1",
    "description": "<what this subtask answers, semantically rich, 1 sentence>",
    "aspect": "<event|spatial|entity|causal|temporal|process>",
    "importance": <float 0-1>,
    "modalities": ["doc_text"]
  },
  {
    "id": "s2",
    "description": "<cross-modal verification>",
    "aspect": "entity",
    "importance": 0.8,
    "modalities": ["doc_text", "video_text"]
  }
]
"""


def run(query: str, provider: Any) -> Optional[List[dict]]:
    """Execute the decompose skill. Returns list of subtask dicts, or None
    on any failure (skill never throws; core loop degrades gracefully).

    Args:
        query: User's question to decompose
        provider: Must have .complete(messages, system, max_tokens) method.
                  ScriptedProvider/MockProvider return None (use real LLM).

    Returns:
        List of subtask dicts with keys: id, description, aspect, importance,
        modalities. Each modalities list has exactly ONE element.
    """
    if provider is None or not hasattr(provider, "complete"):
        return None
    provider_type = type(provider).__name__
    if provider_type in ("ScriptedProvider", "MockProvider"):
        return None

    user_msg = f"QUERY: {query}\n\nDecompose this into 1-5 subtasks."
    try:
        raw = provider.complete(
            messages=[{"role": "user", "content": user_msg}],
            system=_DECOMPOSE_SKILL_SYSTEM,
            max_tokens=1200)  # Increased for richer subtask descriptions
    except Exception as e:
        # Skill failure is non-fatal; core loop degrades gracefully
        return None

    if not raw:
        return None

    subtasks = _extract_json_array(raw)
    if subtasks is None:
        return None

    # Validate: check structure but DON'T filter multi-modality subtasks
    # Orchestrator._pre_run_decompose will split them into single-modality siblings
    valid = []
    for st in subtasks:
        if isinstance(st, dict) and "modalities" in st:
            mods = st.get("modalities", [])
            if isinstance(mods, list) and len(mods) >= 1:
                # Has at least one modality - let orchestrator handle splitting
                valid.append(st)

    return valid if valid else None


def _extract_json_array(text: str) -> Optional[List[dict]]:
    """Tolerant JSON array extractor — strips code fences and locates the
    first [...] block."""
    import json
    import re
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", t).strip()
    m = re.search(r"\[.*\]", t, re.DOTALL)
    if not m:
        return None
    try:
        arr = json.loads(m.group(0))
        return arr if isinstance(arr, list) else None
    except Exception:
        return None
