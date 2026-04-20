"""Wrapper for MDocAgent (Han et al., 2025) — your main competitor.

You need to:
  1. git clone https://github.com/aiming-lab/MDocAgent
  2. pip install their requirements
  3. Set MDOCAGENT_PATH env var or edit _MDOCAGENT_ROOT below
  4. Wire your provider classes into MDocAgent's config/openai.yaml

This wrapper implements the common `run(query, store, providers) -> dict`
interface so `scripts/run_experiments.py` can treat MDocAgent as just
another variant for apples-to-apples comparison.

Note: MDocAgent is VERY similar to MERP in spirit (5 specialist agents,
text+image RAG). The key differences you want to highlight:
  - MDocAgent has NO VoI gate → spends ~2x tokens
  - MDocAgent has NO saturation detection → gets stuck on hard queries
  - MDocAgent has NO cross-modal conflict resolution → inconsistent
    answers when text and image disagree
  - MDocAgent does NOT handle video
  - MDocAgent has NO cross-corpus retrieval

Keep this wrapper FAITHFUL — don't hobble MDocAgent. The comparison
only works if MDocAgent runs at its best on doc-only, and you beat it
on efficiency / cross-corpus.
"""
from __future__ import annotations
import os
import sys
from typing import Any, Dict


_MDOCAGENT_ROOT = os.environ.get("MDOCAGENT_PATH", "")


def _ensure_path():
    if not _MDOCAGENT_ROOT:
        raise ImportError(
            "MDOCAGENT_PATH env var not set. "
            "Clone https://github.com/aiming-lab/MDocAgent and set:\n"
            "  export MDOCAGENT_PATH=/path/to/MDocAgent")
    if _MDOCAGENT_ROOT not in sys.path:
        sys.path.insert(0, _MDOCAGENT_ROOT)


def run(query: str, store, providers) -> Dict[str, Any]:
    """Run MDocAgent on one query using our ingested store.

    MDocAgent assumes a particular dataset layout. The cleanest integration
    path is:
      - Use MDocAgent's retrieval (ColBERT + ColPali) directly on the
        raw PDF files (not our MultiIndexStore)
      - Pass the question + retrieved context through their 5-agent chain
      - Extract the summarizer agent's final answer

    This is a stub that you finish after cloning their repo. For paper-
    comparable numbers, reuse their exact retrieval + agent stack,
    swapping only the LLM provider to match your DA's backbone for
    fair comparison.
    """
    _ensure_path()

    # Adapter shape — FILL IN after cloning MDocAgent:
    #
    # from mdocagent.pipeline import run_query
    # answer = run_query(
    #     question=query,
    #     pdf_id=<derived from the active doc context>,
    #     config={"model": providers["da"].model_name},
    # )
    # return {"answer": answer, "confidence": 0.5,
    #         "reason": "mdocagent_default",
    #         "n_retrieved": 4, "n_tokens": <estimate>}

    raise NotImplementedError(
        "Finish this wrapper by reading MDocAgent's README and filling "
        "in `run_query` import + calling sequence. This is maybe 30 "
        "lines of code once you have their repo cloned.")
