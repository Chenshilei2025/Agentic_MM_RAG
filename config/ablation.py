"""Ablation configuration.

Central toggles for feature ablation studies. Pass an AblationConfig to the
Orchestrator at construction time; it propagates to sub-components via the
factory and the command dispatch branch.

Typical ablation matrix:
  baseline             : everything ON
  no_rerank            : disable rerank_text / rerank_visual / dedupe
  no_tier3_gate        : allow tier-3 writes without REQUEST_FULL_EVIDENCE
  no_parallelism       : serialize sub-agent execution (max_workers=1)
  no_progressive_disc  : sub-agents write summary immediately, no intent step
  combined             : all above disabled
"""
from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class AblationConfig:
    # Rerank/dedupe tools available to sub-agents.
    enable_rerank: bool = True
    # Tier-3 (full) evidence requires prior REQUEST_FULL_EVIDENCE authorization.
    enable_tier3_gate: bool = True
    # Parallel sub-agent execution via WorkerPool.
    enable_parallelism: bool = True
    # Progressive 3-tier disclosure (intent → summary → full). If disabled,
    # sub-agents skip intent and write summary on first step.
    enable_progressive_disclosure: bool = True
    # Deterministic stopping threshold (DA may still override).
    confidence_threshold: float = 0.9
    coverage_threshold: float = 0.8

    # ---- Named presets for easy CLI selection ----
    @classmethod
    def preset(cls, name: str) -> "AblationConfig":
        presets: Dict[str, Dict[str, Any]] = {
            "baseline":            {},
            "no_rerank":           {"enable_rerank": False},
            "no_tier3_gate":       {"enable_tier3_gate": False},
            "no_parallelism":      {"enable_parallelism": False},
            "no_progressive_disc": {"enable_progressive_disclosure": False},
            "combined":            {
                "enable_rerank": False,
                "enable_tier3_gate": False,
                "enable_parallelism": False,
                "enable_progressive_disclosure": False,
            },
        }
        if name not in presets:
            raise ValueError(
                f"unknown ablation preset: {name}. "
                f"choose one of: {list(presets)}")
        return cls(**presets[name])

    def as_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}
