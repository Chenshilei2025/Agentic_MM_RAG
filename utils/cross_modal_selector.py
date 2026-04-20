"""Cross-modal evidence selector for Tier-2 curated disclosure.

Core Design for EMNLP paper:
  - When DA requests curated evidence, select HIGH-VALUE candidates
  - Prioritize: (1) cross-modal conflicts, (2) important+low-confidence
  - Return curated set with cross-modal alignment metadata

This selector runs on the Orchestrator side when processing
REQUEST_CURATED_EVIDENCE commands.
"""
from typing import Any, Dict, List, Optional, Set
import logging

log = logging.getLogger("selector")


class CrossModalSelector:
    """Selects high-value evidence for Tier-2 curated disclosure.

    The selector's job is to identify which evidence candidates deserve
    DA attention, given limited token budget. It prioritizes:

    1. CONFLICT EVIDENCE — candidates from disagreeing modalities
    2. HIGH-VALUE LOW-CONF — important subtasks with low confidence
    3. TOP-RANKED — best candidates per (aspect, modality) pair

    The output is a curated set that enables cross-modal comparison
    without overwhelming the DA context.
    """

    def __init__(self, max_per_agent: int = 10,
                 conflict_boost: float = 2.0):
        """Initialize selector.

        Args:
            max_per_agent: Maximum candidates to return per agent
            conflict_boost: Score multiplier for conflict evidence
        """
        self.max_per_agent = max_per_agent
        self.conflict_boost = conflict_boost

    # ------------------------------------------------------------------
    # Main selection API
    # ------------------------------------------------------------------
    def select_curated(self,
                      agent_id: str,
                      retrieved_candidates: List[Dict[str, Any]],
                      status_snapshot: Dict[str, Any],
                      conflict_aspects: Optional[Set[str]] = None,
                      is_important: bool = False,
                      is_low_confidence: bool = False) -> List[Dict[str, Any]]:
        """Select high-value candidates for curated disclosure.

        Args:
            agent_id: Sub-agent ID
            retrieved_candidates: Raw candidates from retrieval (top 20)
            status_snapshot: Agent's status board snapshot
            conflict_aspects: Aspects with cross-modal disagreement
            is_important: Is this subtask importance >= 0.7?
            is_low_confidence: Is confidence < 0.6?

        Returns:
            Curated list of candidates, sorted by cross-modal value.
            Each entry has: id, relevance, evidence_hit, text, note,
            plus meta from retrieval.
        """
        if not retrieved_candidates:
            return []

        # Scoring: base relevance + context boost
        scored = []
        for c in retrieved_candidates:
            base_score = float(c.get("score", c.get("relevance", 0.0)))
            boost = self._compute_boost(
                c, status_snapshot, conflict_aspects,
                is_important, is_low_confidence
            )
            final_score = base_score * boost
            scored.append({**c, "_selection_score": final_score})

        # Sort by selection score
        scored.sort(key=lambda x: x.get("_selection_score", 0.0), reverse=True)

        # Apply per-agent cap
        curated = scored[:self.max_per_agent]

        # Add cross-modal alignment metadata
        for c in curated:
            c["_selection_metadata"] = {
                "conflict_aspect": (
                    status_snapshot.get("aspect") in conflict_aspects
                    if conflict_aspects else False
                ),
                "important_subtask": is_important,
                "low_confidence_boost": is_low_confidence,
                "original_rank": scored.index(c) + 1
            }

        return curated

    # ------------------------------------------------------------------
    # Boost computation
    # ------------------------------------------------------------------
    def _compute_boost(self,
                       candidate: Dict[str, Any],
                       snapshot: Dict[str, Any],
                       conflict_aspects: Optional[Set[str]],
                       is_important: bool,
                       is_low_confidence: bool) -> float:
        """Compute context-aware boost for a candidate."""
        boost = 1.0

        # Boost 1: Conflict aspect
        if conflict_aspects and snapshot.get("aspect") in conflict_aspects:
            boost *= self.conflict_boost

        # Boost 2: Important + low confidence
        if is_important and is_low_confidence:
            boost *= 1.5

        # Boost 3: Suspicious confidence flag
        if snapshot.get("suspicious"):
            boost *= 1.2

        # Boost 4: High evidence_hit from sub-agent
        evidence_hit = candidate.get("evidence_hit", candidate.get("relevance", 0.5))
        if evidence_hit > 0.8:
            boost *= 1.1

        return boost

    # ------------------------------------------------------------------
    # Cross-modal conflict detection helper
    # ------------------------------------------------------------------
    @staticmethod
    def identify_conflict_aspects(aspect_agreements: List[Dict[str, Any]]) -> Set[str]:
        """Extract aspects with cross-modal disagreement.

        Args:
            aspect_agreements: From EvidencePool.aspect_agreements()

        Returns:
            Set of aspect tags where agreement_state = "disagree"
        """
        conflicts = set()
        for agr in (aspect_agreements or []):
            if agr.get("agreement_state") == "disagree":
                conflicts.add(agr.get("aspect", ""))
        return conflicts

    # ------------------------------------------------------------------
    # Important low-confidence detection helper
    # ------------------------------------------------------------------
    @staticmethod
    def is_important_low_confidence(snapshot: Dict[str, Any]) -> bool:
        """Check if agent is important but low confidence."""
        # Check confidence across all axes
        confs = [
            snapshot.get("confidence", {}).get("retrieval_score"),
            snapshot.get("confidence", {}).get("evidence_agreement"),
            snapshot.get("confidence", {}).get("coverage")
        ]
        max_conf = max((c for c in confs if c is not None), default=0.0)

        # Check coverage
        coverage = snapshot.get("coverage", {}).get("covered", 1.0)

        return (max_conf < 0.6) or (coverage < 0.5)

    # ------------------------------------------------------------------
    # Utility: format curated for sub-agent consumption
    # ------------------------------------------------------------------
    @staticmethod
    def format_for_subagent(curated: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Format curated candidates for sub-agent's curated write.

        Converts internal selection format to the schema expected by
        the write_curated tool.
        """
        formatted = []
        for c in curated:
            entry = {
                "id": c.get("id", ""),
                "relevance": float(c.get("_selection_score",
                                         c.get("relevance", 0.0))),
                "evidence_hit": float(c.get("evidence_hit",
                                           c.get("relevance", 0.5))),
            }
            # Add text/note if available
            if "text" in c:
                entry["text"] = str(c["text"])[:400]
            if "note" in c:
                entry["note"] = str(c["note"])[:300]

            # Add raw if present (subject to authorization)
            if "content" in c or "raw" in c:
                entry["raw"] = str(c.get("content", c.get("raw", "")))[:60_000]

            formatted.append(entry)

        return {"key_candidates": formatted, "with_raw": any("raw" in e for e in formatted)}


# Singleton instance for Orchestrator use
_default_selector = CrossModalSelector()


def select_curated_evidence(*args, **kwargs) -> List[Dict[str, Any]]:
    """Convenience wrapper using default selector."""
    return _default_selector.select_curated(*args, **kwargs)
