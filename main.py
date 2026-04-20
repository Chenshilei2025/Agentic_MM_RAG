"""Entrypoint. Wires providers, sub-agent factory, store, and orchestrator.

Demonstrates the command-driven flow:
  DecisionAgent -> COMMAND -> Orchestrator -> (tool dispatch or topology change)
"""
import argparse, json, os, re, sys, yaml

import tools.builtin  # registers retrieval_text, retrieval_visual,
                       # write_evidence, read_evidence (Round 3 tool set)

from providers.base_provider import ProviderRegistry
from providers.anthropic_provider import AnthropicProvider
from providers.mock_provider import ScriptedProvider
from agents.decision_agent import DecisionAgent
from agents.factory import SubAgentFactory
from memory.store import MultiIndexStore, VectorIndex  # noqa: F401
from orchestrator.controller import Orchestrator
from utils.logger import get_logger

log = get_logger("main")

SAMPLE_INDICES = {
    "doc_text": [
        {"id": "doc_t_1",
         "content": "The Orchestrator is the only executor of tools.",
         "meta": {"source": "design_spec.pdf#section_2"}},
        {"id": "doc_t_2",
         "content": "Agents output structured JSON and never call APIs directly.",
         "meta": {"source": "design_spec.pdf#section_3"}},
    ],
    "doc_visual": [
        {"id": "doc_v_1",
         "content": "Architecture diagram: decision agent on top, orchestrator middle, sub-agents bottom.",
         "meta": {"source": "design_spec.pdf#figure_1", "type": "diagram"}},
    ],
    "video_text": [
        {"id": "vid_t_1",
         "content": "Speaker explains that the orchestrator manages sub-agent lifecycle.",
         "meta": {"source": "talk_v1.mp4", "timestamp": "00:05:12"}},
    ],
    "video_visual": [
        {"id": "vid_v_1",
         "content": "Slide shown: boxes labeled Decision Agent, Orchestrator, SubAgents.",
         "meta": {"source": "talk_v1.mp4", "frame": 1830}},
    ],
}


def _build_sample_store():
    from memory.store import VectorIndex, MultiIndexStore
    indices = {m: VectorIndex(modality=m, docs=docs)
               for m, docs in SAMPLE_INDICES.items()}
    return MultiIndexStore(indices)


def _extract_recent_spawn(blob: str) -> str:
    """Parse a recently-spawned agent id out of the rendered user prompt."""
    # Note: when messages are str()ed for inspection, newlines become the
    # literal two chars "\n", so we don't rely on \s here.
    m = re.search(r"RECENTLY SPAWNED.*?(agent-[0-9a-f]+)", blob, re.DOTALL)
    if m: return m.group(1)
    m = re.search(r"(agent-[0-9a-f]+)", blob)
    return m.group(1) if m else ""


def build_scripted_provider() -> ScriptedProvider:
    """Scripted demo: TWO parallel sub-agents (doc_text + video_text) each
    autonomously push tier-1 (intent) + tier-2 (summary), then DA requests
    tier-3 (full) from one of them, then DA stops.

    Call sequence the provider sees:
       1  DA          SPAWN_AGENTS(doc_text, video_text)
       2  SubA        write_evidence(intent)
       3  SubA        retrieval(doc_text)
       4  SubA        write_evidence(summary)
       5  SubB        write_evidence(intent)
       6  SubB        retrieval(video_text)
       7  SubB        write_evidence(summary)
       8  DA          REQUEST_FULL_EVIDENCE(<first_agent>)
       9  SubA        write_evidence(full)
      10  DA          STOP_TASK

    Parallel sub-agents share the call counter but produce independent
    tier payloads — the helper emits the right payload shape for whichever
    tier/modality that sub-agent is currently in.
    """
    # Per-agent inner step counter. Worker threads run concurrently, so we
    # return the script entry appropriate for THIS agent's progress — not
    # a global counter.
    from threading import Lock
    lock = Lock()
    # Track how many times each call happened from the DA side.
    da_calls = {"n": 0}
    # Track tier progression per agent-id (parsed from system prompt).
    sub_progress: dict[str, int] = {}

    SUB_PAYLOADS = {
        "doc_text": [
            '{"tool_name":"write_evidence","arguments":'
            '{"agent_id":"__self__","stage":"intent","payload":'
            '{"modality":"doc_text","data_source":"design_docs",'
            '"planned_k":3}}}',
            '{"tool_name":"retrieval","arguments":'
            '{"query":"orchestrator executor tools","modality":"doc_text","k":3}}',
            '{"tool_name":"write_evidence","arguments":'
            '{"agent_id":"__self__","stage":"summary","payload":'
            '{"finding":"Orchestrator is sole tool executor (doc_text).",'
            '"coverage":{"covered":0.9,"gaps":[]},'
            '"confidence":{"retrieval_score":"high","evidence_agreement":"high","coverage":"high"},'
            '"n_retrieved":3,"n_kept":2,"top_score":0.88,"score_spread":0.05,'
            '"caveat":null,"citations":["dt_01","dt_02"]}}}',
            '{"tool_name":"write_evidence","arguments":'
            '{"agent_id":"__self__","stage":"full","payload":'
            '{"content":"The Orchestrator is the only executor of tools.",'
            '"sources":["design_spec.pdf#section_2"]}}}',
        ],
        "video_text": [
            '{"tool_name":"write_evidence","arguments":'
            '{"agent_id":"__self__","stage":"intent","payload":'
            '{"modality":"video_text","data_source":"talk_transcripts",'
            '"planned_k":3}}}',
            '{"tool_name":"retrieval","arguments":'
            '{"query":"orchestrator sub-agent lifecycle","modality":"video_text","k":3}}',
            '{"tool_name":"write_evidence","arguments":'
            '{"agent_id":"__self__","stage":"summary","payload":'
            '{"finding":"Speaker confirms orchestrator manages lifecycle.",'
            '"coverage":{"covered":0.85,"gaps":[]},'
            '"confidence":{"retrieval_score":"high","evidence_agreement":"medium","coverage":"high"},'
            '"n_retrieved":3,"n_kept":2,"top_score":0.82,"score_spread":0.08,'
            '"caveat":null,"citations":["vt_01","vt_02"]}}}',
        ],
    }

    def dispatch(messages, system):
        """Route each provider call to either DA or a SubAgent based on the
        rendered prompt. The charter string identifies the caller role."""
        blob = str(messages) + system
        is_sub = "You are a SUB-AGENT" in system
        if is_sub:
            # Identify modality from the injected context.
            m = re.search(r"(doc_text|doc_visual|video_text|video_visual)", blob)
            modality = m.group(1) if m else "doc_text"
            # Identify this sub-agent's id so we can track its tier progress.
            # Sub-agent prompt includes no id directly, but STAGE: intent/summary/full.
            stage_m = re.search(r"STAGE:\s*(\w+)", blob)
            stage = stage_m.group(1) if stage_m else "intent"
            # Use modality+stage as a lookup to the scripted payload.
            with lock:
                idx = sub_progress.get(modality, 0)
                sub_progress[modality] = idx + 1
            # stage index: intent=0, summary=1, full=2 -> but retrieval is
            # a separate step between intent and summary in our script.
            table = SUB_PAYLOADS[modality]
            out = table[min(idx, len(table) - 1)]
            return out

        # Otherwise DA.
        with lock:
            da_calls["n"] += 1
            i = da_calls["n"]
        if i == 1:
            return ('{"command":"SPAWN_AGENTS","arguments":{"specs":['
                    '{"agent_type":"seeker_inspector","modality":"doc_text",'
                    '"goal":"find doc evidence on orchestrator role"},'
                    '{"agent_type":"seeker_inspector","modality":"video_text",'
                    '"goal":"find video evidence on orchestrator role"}'
                    ']}}')
        if i == 2:
            # Authorize the first agent to produce tier-3.
            aid = _extract_recent_spawn(blob)
            # Fall back: grab any agent-id from the prompt
            if not aid:
                ids = re.findall(r"agent-[0-9a-f]+", blob)
                aid = ids[0] if ids else ""
            return ('{"command":"REQUEST_FULL_EVIDENCE","arguments":'
                    f'{{"agent_id":"{aid}"}}}}')
        # Final: stop.
        return ('{"command":"STOP_TASK","arguments":'
                '{"reason":"thresholds met and tier-3 secured",'
                '"answer":"The Orchestrator is the sole tool executor, '
                'corroborated by both document text and video transcript.",'
                '"confidence":0.93}}')

    return ScriptedProvider([dispatch] * 40)  # callable reused for each call


def load_config(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def build_provider(kind: str, cfg: dict):
    if kind == "anthropic":
        model = cfg.get("providers", {}).get("anthropic", {}).get(
            "model", "claude-sonnet-4-5")
        return AnthropicProvider(model=model)
    if kind == "scripted":
        return build_scripted_provider()
    raise ValueError(f"unknown provider: {kind}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--provider", default="scripted",
                    choices=["anthropic", "scripted"])
    ap.add_argument("--config", default="config/system.yaml")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument("--step-timeout", type=float, default=30.0)
    ap.add_argument("--ablation", default="baseline",
                    choices=["baseline", "no_rerank", "no_tier3_gate",
                             "no_parallelism", "no_progressive_disc",
                             "combined"],
                    help="Ablation preset — disables one or more features")
    args = ap.parse_args()

    from config.ablation import AblationConfig
    ablation = AblationConfig.preset(args.ablation)
    log.info(f"[main] ablation={args.ablation} {ablation.as_dict()}")

    cfg = load_config(args.config)
    provider = build_provider(args.provider, cfg)
    ProviderRegistry.register(provider)

    decision = DecisionAgent(provider=provider)
    factory = SubAgentFactory(provider=provider,
                              enable_rerank=ablation.enable_rerank)
    store = _build_sample_store()

    orch = Orchestrator(
        decision_agent=decision, factory=factory, store=store,
        max_steps=args.max_steps or cfg.get("system", {}).get("max_steps", 40),
        max_workers=args.max_workers,
        step_timeout=args.step_timeout,
        ablation=ablation,
    )

    result = orch.run(args.query)
    log.info("=" * 60)
    print(json.dumps(result, indent=2))
    # Surface live-API cost if running Anthropic
    if hasattr(provider, "usage_summary"):
        log.info(f"[usage] {provider.usage_summary()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
