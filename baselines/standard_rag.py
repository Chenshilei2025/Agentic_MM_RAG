"""Standard RAG — simplest baseline.

Single text retrieve → concatenate top-K docs → generate answer.
NO modality awareness, NO decomposition, NO multi-agent.

This is the floor reference: if MERP can't beat this on MADQA,
the paper is in trouble.

Usage:
    from baselines.standard_rag import run
    result = run(query, store, provider, k=5)
"""
from __future__ import annotations
from typing import Any, Dict


SYSTEM_PROMPT = (
    "You are a QA assistant. Answer the user's question based on the "
    "provided document excerpts. Be concise. If the answer is not in the "
    "excerpts, say 'I don't know'. Output JSON: "
    "{\"answer\": str, \"confidence\": float, \"reason\": str}")


def run(query: str, store, provider,
        k: int = 5,
        modality: str = "doc_text") -> Dict[str, Any]:
    """Standard RAG run. Retrieves from ONE modality index and generates.

    Args:
      query     the user question
      store     MultiIndexStore
      provider  LLM with .complete(messages, system=..., max_tokens=...)
      k         top-k to retrieve
      modality  which modality to search (default doc_text)

    Returns:
      {"answer", "confidence", "reason", "n_retrieved", "n_tokens"}
    """
    hits = store.search(query=query, modality=modality, k=k)
    context = _format_hits(hits)
    user_msg = f"QUESTION: {query}\n\nEXCERPTS:\n{context}\n\nAnswer:"
    try:
        raw = provider.complete(
            messages=[{"role": "user", "content": user_msg}],
            system=SYSTEM_PROMPT, max_tokens=400)
    except Exception as e:
        return {"answer": "", "confidence": 0.0,
                "reason": f"provider_error: {e}",
                "n_retrieved": len(hits), "n_tokens": 0}
    parsed = _parse_json_answer(raw)
    parsed["n_retrieved"] = len(hits)
    parsed["n_tokens"] = len(user_msg) // 3      # rough estimate (chars/3)
    return parsed


def _format_hits(hits):
    if not hits:
        return "(no relevant excerpts found)"
    lines = []
    for i, h in enumerate(hits, 1):
        meta = h.get("meta") or {}
        src = meta.get("source", "?")
        page = meta.get("page", "")
        page_str = f" p.{page}" if page else ""
        content = (h.get("content") or "")[:500]
        lines.append(f"[{i}] ({src}{page_str}) {content}")
    return "\n".join(lines)


def _parse_json_answer(raw: str) -> Dict[str, Any]:
    import json, re
    if not raw:
        return {"answer": "", "confidence": 0.0, "reason": "empty"}
    t = raw.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", t).strip()
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if not m:
        return {"answer": t[:300], "confidence": 0.5,
                "reason": "unstructured"}
    try:
        obj = json.loads(m.group(0))
        return {
            "answer": str(obj.get("answer", "")),
            "confidence": float(obj.get("confidence", 0.5)),
            "reason": str(obj.get("reason", "")),
        }
    except Exception:
        return {"answer": t[:300], "confidence": 0.5,
                "reason": "parse_failed"}
