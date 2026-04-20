"""Late-fusion RAG — retrieve each modality independently, concatenate
results, let the LLM fuse.

Stronger than standard_rag because the LLM sees BOTH text and visual
(caption) evidence. Closer to MERP in information available, but WITHOUT
decomposition / reflection / cross-modal conflict resolution.

This is the MOST IMPORTANT baseline for C1 ("Modality-Isolated Retrieval"):
late-fusion has modality-isolated retrieval too, but skips the decision-
layer fusion. If MERP beats late_fusion, C1 is validated.
"""
from __future__ import annotations
from typing import Any, Dict, List

from baselines.standard_rag import (SYSTEM_PROMPT, _format_hits,
                                     _parse_json_answer)


LATE_FUSION_SYSTEM = (
    "You are a multimodal QA assistant. You will see evidence excerpts "
    "from up to 4 modality channels (doc_text, doc_visual captions, "
    "video_text transcripts, video_visual captions). Synthesize a "
    "single answer that uses the best evidence across modalities. "
    "Output JSON: {\"answer\": str, \"confidence\": float, \"reason\": str}")


def run(query: str, store, provider,
        k_per_modality: int = 3,
        modalities: List[str] = None) -> Dict[str, Any]:
    """Retrieve `k_per_modality` from each modality; stack; generate.

    Token cost roughly scales as 4*k_per_modality (vs. single-modality
    standard_rag's k). We set the default k lower (3) to match standard
    budget ceiling.
    """
    modalities = modalities or ["doc_text", "doc_visual",
                                "video_text", "video_visual"]
    blocks = []
    total_hits = 0
    for m in modalities:
        try:
            hits = store.search(query=query, modality=m, k=k_per_modality)
        except KeyError:
            continue
        if not hits:
            continue
        total_hits += len(hits)
        blocks.append(f"--- {m.upper()} ---\n{_format_hits(hits)}")
    context = "\n\n".join(blocks) if blocks else "(no retrieval hits)"
    user_msg = f"QUESTION: {query}\n\nEVIDENCE:\n{context}\n\nAnswer:"
    try:
        raw = provider.complete(
            messages=[{"role": "user", "content": user_msg}],
            system=LATE_FUSION_SYSTEM, max_tokens=500)
    except Exception as e:
        return {"answer": "", "confidence": 0.0,
                "reason": f"provider_error: {e}",
                "n_retrieved": total_hits, "n_tokens": 0}
    parsed = _parse_json_answer(raw)
    parsed["n_retrieved"] = total_hits
    parsed["n_tokens"] = len(user_msg) // 3
    return parsed
