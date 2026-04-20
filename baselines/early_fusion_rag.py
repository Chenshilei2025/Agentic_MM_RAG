"""Early-fusion RAG — merge text + visual documents into ONE index
BEFORE retrieval. Contrasts with late_fusion (separate indices, merge at
generation) and MERP (separate indices, fuse at decision layer).

This implementation does "virtual early fusion": we don't actually
re-embed everything into a shared space; we retrieve from all modality
indices at once and rank the union by score. Close enough for the ablation
purpose and cheaper to run.

This is the key contrast for C1: if MERP beats early-fusion but not by
much, C1's contribution is marginal; if MERP beats it meaningfully, we
have strong evidence for modality-isolated reasoning.
"""
from __future__ import annotations
from typing import Any, Dict, List

from baselines.standard_rag import (_format_hits, _parse_json_answer)


EARLY_FUSION_SYSTEM = (
    "You are a QA assistant. Answer the user's question using the "
    "provided excerpts, which may include text passages and image/chart "
    "captions from documents and videos. Output JSON: "
    "{\"answer\": str, \"confidence\": float, \"reason\": str}")


def run(query: str, store, provider,
        k: int = 8,
        modalities: List[str] = None) -> Dict[str, Any]:
    """Early-fusion: retrieve top-k from each modality, union-rank by score,
    feed top-k to LLM.

    Args:
      k           total number of excerpts to feed (NOT per-modality)
      modalities  which indices to search
    """
    modalities = modalities or ["doc_text", "doc_visual",
                                "video_text", "video_visual"]
    all_hits = []
    for m in modalities:
        try:
            hits = store.search(query=query, modality=m, k=k)
        except KeyError:
            continue
        for h in hits:
            h.setdefault("modality", m)
            all_hits.append(h)
    # Re-rank union by score
    all_hits.sort(key=lambda h: h.get("score", 0), reverse=True)
    top = all_hits[:k]
    context = _format_hits_with_modality(top)
    user_msg = f"QUESTION: {query}\n\nEXCERPTS:\n{context}\n\nAnswer:"
    try:
        raw = provider.complete(
            messages=[{"role": "user", "content": user_msg}],
            system=EARLY_FUSION_SYSTEM, max_tokens=500)
    except Exception as e:
        return {"answer": "", "confidence": 0.0,
                "reason": f"provider_error: {e}",
                "n_retrieved": len(top), "n_tokens": 0}
    parsed = _parse_json_answer(raw)
    parsed["n_retrieved"] = len(top)
    parsed["n_tokens"] = len(user_msg) // 3
    return parsed


def _format_hits_with_modality(hits):
    lines = []
    for i, h in enumerate(hits, 1):
        meta = h.get("meta") or {}
        mod = h.get("modality", "?")
        src = meta.get("source", "?")
        page = meta.get("page", "")
        page_str = f" p.{page}" if page else ""
        content = (h.get("content") or "")[:500]
        lines.append(f"[{i}] ({mod}, {src}{page_str}) {content}")
    return "\n".join(lines) if lines else "(empty)"
