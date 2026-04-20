"""SubAgent factory — Round 3 with per-modality provider routing +
modality-specialized tool whitelists.

Provider routing precedence:
  1. provider_per_modality[modality]   (exact match; most specific)
  2. provider_visual                   (if modality in {*_visual})
  3. provider                          (fallback)

Tool whitelist (Round 3):
  Text sub-agents:   retrieval_text + write_evidence + read_evidence
  Visual sub-agents: retrieval_visual + write_evidence + read_evidence
  (plus legacy `retrieval` kept for back-compat)

Sub-agent NEVER gets spawn_agent or stop_agent — hard-coded forbidden.
"""
from typing import Dict, List, Optional
from agents.sub_agent import SubAgent

_FORBIDDEN = {"spawn_agent", "stop_agent"}

_TEXT_TOOLS: List[str] = [
    "retrieval_text", "retrieval",
    "write_evidence", "read_evidence",
]
_VISUAL_TOOLS: List[str] = [
    "retrieval_visual", "retrieval",
    "write_evidence", "read_evidence",
]
DEFAULT_SUBAGENT_TOOLS: List[str] = [
    "retrieval", "retrieval_text", "retrieval_visual",
    "write_evidence", "read_evidence",
]
DEFAULT_SUBAGENT_TOOLS_NO_RERANK: List[str] = DEFAULT_SUBAGENT_TOOLS

_VISUAL_MODALITIES = ("doc_visual", "video_visual")


def _tools_for_modality(modality: Optional[str]) -> List[str]:
    if modality in _VISUAL_MODALITIES:
        return list(_VISUAL_TOOLS)
    if modality in ("doc_text", "video_text"):
        return list(_TEXT_TOOLS)
    return list(DEFAULT_SUBAGENT_TOOLS)


class SubAgentFactory:
    def __init__(self, provider, allowed_tools: Optional[List[str]] = None,
                 enable_rerank: bool = True,
                 provider_visual=None,
                 provider_per_modality: Optional[Dict[str, object]] = None):
        if allowed_tools is not None:
            bad = set(allowed_tools) & _FORBIDDEN
            if bad:
                raise ValueError(f"sub-agents must not have tools: {bad}")
            self._allowed_override: Optional[List[str]] = list(allowed_tools)
        else:
            self._allowed_override = None
        self._provider = provider
        self._provider_visual = provider_visual or provider
        self._provider_per_modality = dict(provider_per_modality or {})

    def provider_for(self, modality: Optional[str]):
        if modality and modality in self._provider_per_modality:
            return self._provider_per_modality[modality]
        if modality in _VISUAL_MODALITIES:
            return self._provider_visual
        return self._provider

    def tools_for(self, modality: Optional[str]) -> List[str]:
        if self._allowed_override is not None:
            return list(self._allowed_override)
        return _tools_for_modality(modality)

    def create(self, role: str, task: str,
               modality: Optional[str] = None) -> SubAgent:
        prov = self.provider_for(modality)
        tools = self.tools_for(modality)
        return SubAgent(provider=prov, allowed_tools=tools,
                        role=role, task=task)
