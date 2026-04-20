"""Wrapper for VideoRAG (Ren et al., 2025, KDD 2026) — your video-side
competitor.

You need to:
  1. git clone https://github.com/HKUDS/VideoRAG
  2. pip install their requirements (LightRAG + nano-graphrag deps)
  3. Set VIDEORAG_PATH env var
  4. Wire your provider classes

VideoRAG's dual-channel design (graph-based text grounding + multimodal
context encoding) is a strong baseline for the LongerVideos benchmark.

Key differences you want to highlight:
  - VideoRAG does NOT handle documents
  - VideoRAG has NO explicit cost control
  - VideoRAG has NO cross-corpus retrieval
  - VideoRAG has NO cross-modal conflict handling (it's single-channel
    internally in many pipelines)
"""
from __future__ import annotations
import os
import sys
from typing import Any, Dict


_VIDEORAG_ROOT = os.environ.get("VIDEORAG_PATH", "")


def _ensure_path():
    if not _VIDEORAG_ROOT:
        raise ImportError(
            "VIDEORAG_PATH env var not set. "
            "Clone https://github.com/HKUDS/VideoRAG and set:\n"
            "  export VIDEORAG_PATH=/path/to/VideoRAG")
    if _VIDEORAG_ROOT not in sys.path:
        sys.path.insert(0, _VIDEORAG_ROOT)


def run(query: str, store, providers) -> Dict[str, Any]:
    _ensure_path()
    # Finish this wrapper after reading VideoRAG's reproduce scripts
    # (they have a run_videorag.sh that documents the entry point).
    raise NotImplementedError(
        "Finish this wrapper after cloning VideoRAG and pinning a reference"
        " entry point. See VideoRAG-algorithm/reproduce in their repo.")
