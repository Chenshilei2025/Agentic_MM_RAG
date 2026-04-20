"""Sub-agent tool catalog — Round 3 rebuild.

Round 3 change: SPLIT the single `retrieval` tool into TWO modality-
specialized tools. Sub-agent has ONE retrieval tool available (whichever
matches its assigned modality). This means:

  text-class sub-agent    → retrieval_text   → searches {doc_text|video_text}
  visual-class sub-agent  → retrieval_visual → searches {doc_visual|video_visual}

The tool does NOT do rerank, dedupe, or fusion. It returns up to 20 RAW
candidates; the sub-agent (acting as reranker) picks the best ones
manually by READING content, and cites the chosen ids in its summary.

Why not bundle rerank in the tool?
  1. Your core claim: sub-agent IS the reranker. The LLM doing rerank is
     what makes this "agent as reranker". Hiding it in a tool undermines
     that.
  2. Rerank quality is modality-specific (text → cross-encoder; visual →
     CLIP score + LLM re-judging). Sub-agent does the right version for
     its modality.
  3. Dedup is trivially emergent from the sub-agent citation phase —
     picking 5 of 20 candidates means dropping 15.

Tool set (3 tools per sub-agent):
  retrieval_text    OR    retrieval_visual   (one, based on modality)
  write_evidence    (always)
  read_evidence     (rarely used)
"""
from typing import Any, Dict, List
from tools.registry import tool
from memory.evidence_pool import EvidencePool


# Default recall count — 10 per Round 3+ spec. Sub-agent reranks down to ~5.
DEFAULT_RECALL_K = 10


def _resolve_text_modality(ctx: Dict[str, Any]) -> str:
    """Pick doc_text vs video_text from the sub-agent's assigned modality."""
    m = (ctx.get("agent_modality") or "").strip()
    if m in ("doc_text", "video_text"):
        return m
    # Sub-agent is text-class but modality not set — default to doc_text.
    return "doc_text"


def _resolve_visual_modality(ctx: Dict[str, Any]) -> str:
    m = (ctx.get("agent_modality") or "").strip()
    if m in ("doc_visual", "video_visual"):
        return m
    return "doc_visual"


@tool("retrieval_text",
      "Search YOUR assigned TEXT modality index (doc_text or video_text). "
      "Returns up to `top_k` RAW candidates as {id, content, score, meta}. "
      "This is a vector-match recall only — NO rerank, NO dedupe. "
      "Your job after this call is to READ all returned candidates, "
      "mentally rerank them by relevance to your goal, drop near-duplicates, "
      "and cite the top ids in write_evidence(stage='summary'). "
      "You are the reranker.",
      {"type": "object",
       "properties": {
           "query":    {"type": "string",
                        "description": "natural-language search query using "
                                       "the key phrases from your goal"},
           "top_k":    {"type": "integer", "default": DEFAULT_RECALL_K,
                        "description": f"default {DEFAULT_RECALL_K}; "
                                       f"use default unless you truly need less"},
           "exclude_ids": {"type": "array",
                           "items": {"type": "string"},
                           "description": "optional; use on CONTINUE_RETRIEVAL "
                                          "to avoid re-fetching already-seen ids"},
       },
       "required": ["query"]})
def retrieval_text(query: str,
                   top_k: int = DEFAULT_RECALL_K,
                   exclude_ids: List[str] = None,
                   **ctx) -> List[Dict[str, Any]]:
    store = ctx["store"]
    modality = _resolve_text_modality(ctx)
    raw_k = top_k + len(exclude_ids or [])
    hits = store.search(query=query, modality=modality, k=raw_k)
    if exclude_ids:
        exc = set(str(x) for x in exclude_ids)
        hits = [h for h in hits if str(h.get("id")) not in exc]
    # Tag modality onto each hit so downstream (pool.note_retrieved_candidates
    # + DA status board) knows the source without re-lookup.
    for h in hits:
        h.setdefault("modality", modality)
    return hits[:top_k]


@tool("retrieval_visual",
      "Search YOUR assigned VISUAL modality index (doc_visual or "
      "video_visual). Returns up to `top_k` RAW candidates as "
      "{id, content, score, meta}. `content` is typically a caption / OCR "
      "fragment; `meta.asset_type='image'` and `meta.asset_uri` points to "
      "the actual image. NO rerank, NO dedupe. Your job is to READ the "
      "captions (and if allowed, the images), mentally rerank them by "
      "relevance to your visual goal, drop near-duplicates, and cite the "
      "top ids in write_evidence(stage='summary').",
      {"type": "object",
       "properties": {
           "query":    {"type": "string",
                        "description": "natural-language description of the "
                                       "visual content you want to find"},
           "top_k":    {"type": "integer", "default": DEFAULT_RECALL_K,
                        "description": f"default {DEFAULT_RECALL_K}"},
           "exclude_ids": {"type": "array",
                           "items": {"type": "string"},
                           "description": "optional; used on CONTINUE_RETRIEVAL"},
       },
       "required": ["query"]})
def retrieval_visual(query: str,
                     top_k: int = DEFAULT_RECALL_K,
                     exclude_ids: List[str] = None,
                     **ctx) -> List[Dict[str, Any]]:
    store = ctx["store"]
    modality = _resolve_visual_modality(ctx)
    raw_k = top_k + len(exclude_ids or [])
    hits = store.search(query=query, modality=modality, k=raw_k)
    if exclude_ids:
        exc = set(str(x) for x in exclude_ids)
        hits = [h for h in hits if str(h.get("id")) not in exc]
    for h in hits:
        h.setdefault("modality", modality)
    return hits[:top_k]


# Back-compat: legacy `retrieval` tool kept as thin dispatcher. Existing
# tests and old sub-agent prompts still call it with an explicit `modality`.
# New Round-3 sub-agents should prefer retrieval_text / retrieval_visual.
@tool("retrieval",
      "[LEGACY] Search a specific modality's index. Prefer retrieval_text "
      "or retrieval_visual when available. Returns {id, content, score, meta}.",
      {"type": "object",
       "properties": {
           "query":    {"type": "string"},
           "modality": {"type": "string",
                        "enum": ["doc_text", "doc_visual",
                                 "video_text", "video_visual"]},
           "top_k":    {"type": "integer", "default": 10},
           "exclude_ids": {"type": "array",
                           "items": {"type": "string"}},
       },
       "required": ["query", "modality"]})
def retrieval(query: str, modality: str,
              top_k: int = 10,
              exclude_ids: List[str] = None,
              **ctx) -> List[Dict[str, Any]]:
    store = ctx["store"]
    raw_k = top_k + len(exclude_ids or [])
    hits = store.search(query=query, modality=modality, k=raw_k)
    if exclude_ids:
        exc = set(str(x) for x in exclude_ids)
        hits = [h for h in hits if str(h.get("id")) not in exc]
    for h in hits:
        h.setdefault("modality", modality)
    return hits[:top_k]


@tool("write_evidence",
      "Record an evidence entry in the pool. Sub-agents call this ONCE per "
      "lifetime for stage='summary' (Tier-1). Additional calls for "
      "stage='sketch' (Tier-2) or stage='curated' (Tier-2) happen only after "
      "DA issues REQUEST_EVIDENCE_SKETCH or REQUEST_CURATED_EVIDENCE. "
      "For curated stage, cross-modal analysis is performed automatically.",
      {"type": "object",
       "properties": {"agent_id": {"type": "string", "description": "use '__self__'"},
                      "stage":    {"type": "string",
                                   "enum": ["intent", "summary", "full",
                                            "sketch", "curated"]},
                      "payload":  {"type": "object"}},
       "required": ["agent_id", "stage", "payload"]})
def write_evidence(agent_id: str, stage: str, payload: Dict[str, Any],
                   **ctx) -> Dict[str, Any]:
    pool: EvidencePool = ctx["pool"]
    # For curated stage, pass aspect_agreements for cross-modal analysis
    aspect_agreements = ctx.get("_aspect_agreements", None)
    eid = pool.write(agent_id=agent_id, stage=stage, payload=payload,
                     aspect_agreements=aspect_agreements)
    return {"evidence_id": eid, "stage": stage}


@tool("read_evidence",
      "Read the status board. Sub-agents rarely need this.",
      {"type": "object",
       "properties": {"include_terminated": {"type": "boolean"}},
       "required": []})
def read_evidence(include_terminated: bool = False, **ctx):
    pool: EvidencePool = ctx["pool"]
    return pool.status_board(include_terminated=include_terminated)
