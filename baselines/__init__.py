"""Baselines for the EMNLP paper. Each baseline provides a `run(query,
store, provider)` → {answer, confidence, reason, n_tokens} function so
the ablation runner can swap them in uniformly.

Baselines implemented:
  standard_rag.py       Single retrieve → single generate, no decomposition
  late_fusion_rag.py    Retrieve independently per modality → concat → generate
  early_fusion_rag.py   Merge text+visual into one index BEFORE retrieval
  self_rag_style.py     Iterative retrieve-and-critique (Self-RAG approximation)

Our MERP system itself lives under `orchestrator/controller.py` — the
ablation runner (scripts/run_ablation.py) treats it as one "variant" among
these baselines for apples-to-apples comparison.

Note: these are INTENTIONALLY simple reference implementations. They're
meant to be strong-enough-to-not-be-strawmen, not hyperparam-tuned SOTA.
If a reviewer challenges "why didn't you tune baseline X harder?" the
honest answer is "this paper's claim is about agentic structure, not raw
retrieval quality; all baselines use the same underlying BGE+CLIP store
and the same LLM".
"""
