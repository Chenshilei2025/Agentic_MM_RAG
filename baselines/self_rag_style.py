"""Self-RAG-style baseline (simplified).

Approximates Asai et al. 2024 "Self-RAG" behaviour:
  1. Generate an initial answer from a first retrieve.
  2. Model emits "reflection tokens" (we simplify: a confidence score + a
     need_more_evidence flag).
  3. If confidence low or need_more_evidence=true, retrieve again with
     reformulated query. Up to N iterations.
  4. Generate final answer.

This is the MOST IMPORTANT baseline for claiming novelty: Self-RAG is
one of the direct predecessors for "adaptive retrieval". If MERP doesn't
beat Self-RAG-style, reviewers will ask "is this just a more expensive
Self-RAG?"

NOT a faithful reimplementation of Self-RAG (which requires a specially
fine-tuned LM). This is an "approximate, prompt-based" variant.
"""
from __future__ import annotations
import json
import re
from typing import Any, Dict, List

from baselines.standard_rag import _format_hits, _parse_json_answer


SELF_RAG_SYSTEM = (
    "You are a QA assistant that retrieves evidence iteratively. On each "
    "turn you read the current evidence and output JSON:\n"
    "{\"answer\": str,        # your best answer so far\n"
    " \"confidence\": float,   # 0-1\n"
    " \"need_more\": bool,     # true if more retrieval would help\n"
    " \"next_query\": str,     # if need_more, what to search for next\n"
    " \"reason\": str}\n"
    "Be concise. If evidence is sufficient, set need_more=false and stop.")


def run(query: str, store, provider,
        k: int = 5,
        modality: str = "doc_text",
        max_iters: int = 3) -> Dict[str, Any]:
    """Iteratively retrieve+generate, up to max_iters rounds.

    Returns:
      {"answer", "confidence", "reason", "n_retrieved", "n_tokens",
       "n_iters": int}
    """
    all_hits: List[dict] = []
    seen_ids = set()
    current_query = query
    total_tokens = 0
    last_result = None

    for it in range(1, max_iters + 1):
        # Retrieve, excluding already-seen
        hits = store.search(query=current_query, modality=modality, k=k)
        new_hits = [h for h in hits if str(h.get("id")) not in seen_ids]
        seen_ids.update(str(h.get("id")) for h in new_hits)
        all_hits.extend(new_hits)

        # Prompt with ALL accumulated evidence (so LLM sees growth)
        context = _format_hits(all_hits[:20])    # cap shown evidence
        user_msg = (f"QUESTION: {query}\n"
                    f"ITERATION: {it}/{max_iters}\n\n"
                    f"EVIDENCE SO FAR:\n{context}\n\n"
                    f"Output JSON.")
        total_tokens += len(user_msg) // 3

        try:
            raw = provider.complete(
                messages=[{"role": "user", "content": user_msg}],
                system=SELF_RAG_SYSTEM, max_tokens=500)
        except Exception as e:
            return {"answer": "", "confidence": 0.0,
                    "reason": f"provider_error: {e}",
                    "n_retrieved": len(all_hits),
                    "n_tokens": total_tokens,
                    "n_iters": it}

        parsed = _parse_self_rag_json(raw)
        last_result = parsed

        if not parsed.get("need_more", False) or it == max_iters:
            break
        # Reformulate query for next iteration
        nxt = (parsed.get("next_query") or "").strip()
        if nxt and nxt != current_query:
            current_query = nxt

    out = {
        "answer": last_result.get("answer", ""),
        "confidence": last_result.get("confidence", 0.0),
        "reason": last_result.get("reason", ""),
        "n_retrieved": len(all_hits),
        "n_tokens": total_tokens,
        "n_iters": it,
    }
    return out


def _parse_self_rag_json(raw: str) -> Dict[str, Any]:
    """Parse both Self-RAG (need_more) and standard (answer-only) JSON."""
    if not raw:
        return {"answer": "", "confidence": 0.0, "need_more": False,
                "reason": "empty"}
    t = raw.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", t).strip()
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if not m:
        return {"answer": t[:300], "confidence": 0.5, "need_more": False,
                "reason": "unstructured"}
    try:
        obj = json.loads(m.group(0))
        return {
            "answer":       str(obj.get("answer", "")),
            "confidence":   float(obj.get("confidence", 0.5)),
            "need_more":    bool(obj.get("need_more", False)),
            "next_query":   str(obj.get("next_query", "")),
            "reason":       str(obj.get("reason", "")),
        }
    except Exception:
        return {"answer": t[:300], "confidence": 0.5, "need_more": False,
                "reason": "parse_failed"}
