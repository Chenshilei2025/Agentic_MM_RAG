#!/usr/bin/env python3
"""End-to-end MADQA live demo.

Flow:
  1. Load MADQA examples + pdf_paths         (data.loaders.madqa)
  2. Parse each PDF → ParsedBlocks           (data.parsers.pdf_parser)
  3. Build MultiIndexStore with BGE + CLIP   (data.index_builder)
  4. For each QA example, run the agentic RAG over the index
  5. Log answer + gold; compute EM/F1 aggregate at the end

Usage:
    python scripts/live_demo_madqa.py \
        --root /data/madqa \
        --limit 20 \
        --da-model claude-sonnet-4 \
        --sub-text-model qwen-7b \
        --sub-visual-model qwen-vl-7b

If you don't have the heavy deps (pdfplumber / BGE / CLIP / etc), pass
--use-mocks to run the skeleton end-to-end with mock embedders and a
scripted provider. That proves the pipeline shape is correct before you
burn GPU time.

Providers are pluggable — the script wires together:
  Decision Agent provider → DA brain (Claude Sonnet / Qwen-72B / DeepSeek-V3)
  Sub-agent text provider → 7B-14B text LLM (Qwen-7B)
  Sub-agent visual provider → VLM (Qwen-VL-7B)

Swap by editing the `_build_providers(...)` function below.
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import sys
import time
from typing import Any, List, Tuple


def parse_args():
    p = argparse.ArgumentParser(description="MADQA live demo")
    p.add_argument("--root", required=True,
                   help="path to MADQA release root")
    p.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    p.add_argument("--limit", type=int, default=20,
                   help="cap on number of QA examples to run")
    p.add_argument("--output", default="madqa_results.jsonl",
                   help="output file for per-example results")
    p.add_argument("--use-mocks", action="store_true",
                   help="use MockBGE + MockCLIPImage + ScriptedProvider; "
                        "good for smoke-testing without GPU/deps")
    p.add_argument("--da-model", default="claude-sonnet-4")
    p.add_argument("--sub-text-model", default="qwen-7b-chat")
    p.add_argument("--sub-visual-model", default="qwen-vl-7b")
    p.add_argument("--max-steps", type=int, default=25)
    p.add_argument("--budget-tokens", type=int, default=30000)
    return p.parse_args()


def _build_providers(args):
    """Return (da_provider, factory) — edit this to plug your actual
    provider objects. All providers must implement `complete(messages,
    system=..., max_tokens=...) -> str`."""
    if args.use_mocks:
        from providers.scripted import ScriptedProvider
        # Single stop-immediately script — just verifies pipeline plumbing.
        scripted_response = ('{"command":"STOP_TASK","arguments":'
                              '{"reason":"mock","answer":"mock answer",'
                              '"confidence":0.5}}')
        da = ScriptedProvider(responses=[scripted_response] * 50)
        from agents.factory import SubAgentFactory
        factory = SubAgentFactory(provider=da)
        return da, factory

    # Real providers — edit these for your account / endpoints.
    # The provider classes you wire here must live under providers/.
    try:
        from providers.anthropic import AnthropicProvider     # type: ignore
        from providers.qwen import QwenProvider               # type: ignore
    except ImportError:
        print("[demo] providers.anthropic or providers.qwen not found. "
              "Either implement them or run with --use-mocks.",
              file=sys.stderr)
        sys.exit(2)

    da = AnthropicProvider(model=args.da_model)
    sub_text = QwenProvider(model=args.sub_text_model)
    sub_visual = QwenProvider(model=args.sub_visual_model, vision=True)

    from agents.factory import SubAgentFactory
    factory = SubAgentFactory(
        provider=sub_text,
        provider_visual=sub_visual,
        provider_per_modality={
            "doc_text":     sub_text,
            "video_text":   sub_text,
            "doc_visual":   sub_visual,
            "video_visual": sub_visual,
        })
    return da, factory


def _build_embedders(args):
    if args.use_mocks:
        from data.embedders.bge_text import MockBGEEmbedder
        from data.embedders.clip_image import MockCLIPImageEmbedder
        return MockBGEEmbedder(), MockCLIPImageEmbedder()
    from data.embedders.bge_text import BGEEmbedder
    from data.embedders.clip_image import CLIPImageEmbedder
    return BGEEmbedder(), CLIPImageEmbedder()


def _ingest_pdfs(pdf_paths: dict,
                 text_emb, image_emb,
                 limit_docs: int = None) -> "MultiIndexStore":
    """Parse + embed all referenced PDFs, return the MultiIndexStore."""
    from data.parsers.pdf_parser import parse_pdf
    from data.index_builder import build_indices

    all_blocks = []
    items = list(pdf_paths.items())
    if limit_docs:
        items = items[:limit_docs]
    for i, (doc_id, path) in enumerate(items):
        t0 = time.time()
        blocks = parse_pdf(path, source_id=doc_id)
        print(f"[demo]   parsed {doc_id}: {len(blocks)} blocks "
              f"({time.time() - t0:.1f}s)")
        all_blocks.extend(blocks)

    print(f"[demo] total blocks: {len(all_blocks)}; building indices...")
    t0 = time.time()
    store = build_indices(all_blocks, text_emb, image_emb)
    print(f"[demo] indices built in {time.time() - t0:.1f}s")
    return store


def run_one(query: str, store, da_provider, factory,
            max_steps: int, budget_tokens: int) -> dict:
    """Run one query through the orchestrator end-to-end."""
    from orchestrator.controller import Orchestrator
    from memory.state_manager import StateManager
    from agents.decision_agent import DecisionAgent
    from config.ablation import AblationConfig

    orch = Orchestrator(
        DecisionAgent(da_provider), factory, store,
        ablation=AblationConfig.preset("baseline"),
        max_steps=max_steps)
    result = orch.run(query)
    return result


def score_example(pred: Any, gold: Any) -> dict:
    """EM + token-F1 — same shape as HotpotQA scorer."""
    if isinstance(pred, dict):
        pred = pred.get("answer", "")
    if isinstance(gold, list):
        gold_list = gold
    else:
        gold_list = [gold]
    pred_str = str(pred or "").lower().strip()
    em = int(any(pred_str == str(g).lower().strip() for g in gold_list))

    def f1(pred_tokens, gold_tokens):
        if not pred_tokens or not gold_tokens:
            return 0.0
        common = set(pred_tokens) & set(gold_tokens)
        if not common:
            return 0.0
        p = len(common) / len(pred_tokens)
        r = len(common) / len(gold_tokens)
        return 2 * p * r / (p + r)
    pred_tokens = pred_str.split()
    best_f1 = max((f1(pred_tokens, str(g).lower().split())
                   for g in gold_list), default=0.0)
    return {"em": em, "f1": best_f1}


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    args = parse_args()
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))

    from data.loaders.madqa import load_madqa

    print(f"[demo] loading MADQA from {args.root} split={args.split}...")
    examples, pdf_paths = load_madqa(args.root, split=args.split,
                                     limit=args.limit)
    print(f"[demo] loaded {len(examples)} QA examples "
          f"over {len(pdf_paths)} pdfs")

    text_emb, image_emb = _build_embedders(args)
    referenced = {e.doc_id for e in examples if e.doc_id in pdf_paths}
    referenced_paths = {d: pdf_paths[d] for d in referenced}
    store = _ingest_pdfs(referenced_paths, text_emb, image_emb)

    da_provider, factory = _build_providers(args)

    # Run each example
    results = []
    scores = {"em": 0.0, "f1": 0.0, "n": 0}
    with open(args.output, "w", encoding="utf-8") as fout:
        for i, ex in enumerate(examples):
            print(f"\n[demo] === QA {i+1}/{len(examples)}  id={ex.id} ===")
            print(f"[demo] Q: {ex.question}")
            t0 = time.time()
            try:
                result = run_one(ex.question, store, da_provider, factory,
                                 args.max_steps, args.budget_tokens)
            except Exception as e:
                logging.exception(f"run failed for {ex.id}")
                result = {"answer": "", "confidence": 0.0,
                          "reason": f"error: {e}"}
            elapsed = time.time() - t0
            metrics = score_example(result.get("answer", ""), ex.answer)
            scores["em"] += metrics["em"]
            scores["f1"] += metrics["f1"]
            scores["n"] += 1

            out_rec = {
                "id": ex.id, "doc_id": ex.doc_id,
                "question": ex.question,
                "gold": ex.answer,
                "pred": result.get("answer", ""),
                "confidence": result.get("confidence"),
                "reason": result.get("reason"),
                "elapsed_sec": round(elapsed, 2),
                "metrics": metrics,
            }
            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            fout.flush()
            results.append(out_rec)
            print(f"[demo] A: {out_rec['pred']}")
            print(f"[demo] gold: {ex.answer}")
            print(f"[demo] EM={metrics['em']} F1={metrics['f1']:.3f} "
                  f"({elapsed:.1f}s)")

    # Aggregate
    n = max(scores["n"], 1)
    print(f"\n[demo] === AGGREGATE ({scores['n']} examples) ===")
    print(f"[demo] EM : {scores['em']/n:.4f}")
    print(f"[demo] F1 : {scores['f1']/n:.4f}")
    print(f"[demo] results written to {args.output}")


if __name__ == "__main__":
    main()
