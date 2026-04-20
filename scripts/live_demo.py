#!/usr/bin/env python
"""Live Anthropic demo — runs the real API end-to-end.

REQUIREMENTS
  pip install anthropic pyyaml
  export ANTHROPIC_API_KEY=sk-ant-...

USAGE
  python scripts/live_demo.py
  python scripts/live_demo.py --query "find visual evidence for X"
  python scripts/live_demo.py --ablation no_rerank
  python scripts/live_demo.py --ablation no_tier3_gate

WHAT IT DOES
  Fires a real Anthropic request for every Decision-Agent turn and every
  sub-agent step. The prompts used are exactly those rendered by
  prompts/prompt_builder.py — so this is also a validation that the charter
  text + payload schemas are clear enough for Claude to follow.

OBSERVABILITY
  - Every tool call is logged by the Orchestrator.
  - Every LLM call logs (input_tokens, output_tokens).
  - Final output prints total usage, so you can cost the run.

COST SANITY CHECK
  A single multi-modal query typically costs:
    Decision Agent: 3-5 calls, ~1500-3000 tokens each  ≈ 10k input tokens
    Sub-agents:     2-3 calls per sub-agent x N agents ≈ 5-15k input tokens
  With Claude Sonnet input price this is cents per run. Tier-3 full writes
  can be large; REQUEST_FULL_EVIDENCE sparingly.
"""
import argparse, json, sys, os

# Make the project root importable when running from scripts/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tools.builtin        # noqa: F401  (registers tools)
try:
    import tools.rerank     # noqa: F401  (legacy, optional)
except ImportError: pass
try:
    import tools.hyde       # noqa: F401  (legacy, optional)
except ImportError: pass

from providers.anthropic_provider import AnthropicProvider
from providers.base_provider import ProviderRegistry
from agents.decision_agent import DecisionAgent
from agents.factory import SubAgentFactory
from memory.store import VectorIndex, MultiIndexStore
from orchestrator.controller import Orchestrator
from config.ablation import AblationConfig
from utils.logger import get_logger

log = get_logger("live")

# Four-index sample corpus for a live multimodal query.
DEMO_CORPUS = {
    "doc_text": [
        {"id": "paper1_abs",
         "content": "We propose a multi-agent RAG system where an orchestrator "
                    "dispatches specialized retrievers in parallel.",
         "meta": {"source": "paper1.pdf#abstract"}},
        {"id": "paper2_intro",
         "content": "Progressive disclosure reduces the reasoning model's "
                    "context by deferring full evidence until requested.",
         "meta": {"source": "paper2.pdf#intro"}},
    ],
    "doc_visual": [
        {"id": "paper1_fig1",
         "content": "Architecture diagram showing decision agent, orchestrator, "
                    "and parallel worker agents.",
         "meta": {"source": "paper1.pdf#fig1", "type": "diagram"}},
    ],
    "video_text": [
        {"id": "talk_clip1",
         "content": "The speaker explains that the orchestrator is a pure "
                    "workflow engine with no LLM calls.",
         "meta": {"source": "talk1.mp4", "timestamp": "00:12:45"}},
    ],
    "video_visual": [
        {"id": "talk_slide7",
         "content": "Slide depicting four retrieval indices: doc_text, "
                    "doc_visual, video_text, video_visual.",
         "meta": {"source": "talk1.mp4", "frame": 4210}},
    ],
}


def build_store() -> MultiIndexStore:
    return MultiIndexStore({
        m: VectorIndex(m, docs) for m, docs in DEMO_CORPUS.items()
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--query", default="Explain how the orchestrator coordinates "
        "sub-agents, with evidence from both documents and videos.")
    ap.add_argument("--ablation", default="baseline",
                    choices=["baseline", "no_rerank", "no_tier3_gate",
                             "no_parallelism", "no_progressive_disc",
                             "combined"])
    ap.add_argument("--max-steps", type=int, default=20,
                    help="turns before forced termination (real LLM "
                         "occasionally emits malformed JSON; keep slack)")
    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument("--step-timeout", type=float, default=60.0)
    ap.add_argument("--model", default=None,
                    help="override ANTHROPIC_MODEL env var")
    ap.add_argument("--dry-run", action="store_true",
                    help="skip API call, print what would have been sent")
    ap.add_argument("--trace-out", default=None,
                    help="write full trace JSONL to this path")
    args = ap.parse_args()

    if args.dry_run:
        # Print the first DA prompt so you can inspect it before paying.
        from prompts.prompt_builder import PromptBuilder
        prompt = PromptBuilder.decision(
            query=args.query, status_board=[], coverage=0.0, max_conf=0.0,
            step=0, recent_spawns=[], archived_agents=[],
            budget={"steps_used": 0, "steps_max": args.max_steps,
                    "tokens_used": 0})
        print("=== SYSTEM PROMPT (first 800 chars) ===")
        print(prompt["system"][:800])
        print("\n=== USER PROMPT (first turn) ===")
        print(prompt["messages"][0]["content"])
        return 0

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set.", file=sys.stderr)
        print("  export ANTHROPIC_API_KEY=sk-ant-...", file=sys.stderr)
        return 2

    ablation = AblationConfig.preset(args.ablation)
    log.info(f"[live] ablation={args.ablation}")

    provider = AnthropicProvider(model=args.model)
    ProviderRegistry.register(provider)

    decision = DecisionAgent(provider=provider)
    factory = SubAgentFactory(provider=provider,
                              enable_rerank=ablation.enable_rerank)
    store = build_store()

    orch = Orchestrator(
        decision_agent=decision, factory=factory, store=store,
        max_steps=args.max_steps, max_workers=args.max_workers,
        step_timeout=args.step_timeout, ablation=ablation,
    )

    log.info("=" * 60)
    log.info(f"QUERY: {args.query}")
    log.info("=" * 60)

    result = orch.run(args.query)

    log.info("=" * 60)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    log.info(f"[usage] {provider.usage_summary()}")

    if args.trace_out:
        with open(args.trace_out, "w", encoding="utf-8") as f:
            for ev in orch.last_state_trace:
                f.write(json.dumps(ev, ensure_ascii=False) + "\n")
            for ev in orch.last_pool_trace:
                f.write(json.dumps({**ev, "_source": "pool"},
                                   ensure_ascii=False, default=str) + "\n")
        log.info(f"[trace] written to {args.trace_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
