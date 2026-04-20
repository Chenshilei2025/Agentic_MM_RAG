"""Decision loop progress tracker — prevents infinite loops and ensures forward progress.

Core Design for EMNLP paper:
  - Monitors each iteration for measurable progress
  - Detects stagnation (no improvement across multiple turns)
  - Enforces "solve problems from previous round" requirement
  - Prevents redundant commands that waste tokens
  - Critical for controlling costs with Claude Opus 4.7

The tracker monitors:
  - New information gained per turn (new aspects covered)
  - Conflict resolution progress
  - Evidence quality improvements
  - Command diversity (avoid repeating failed actions)

When stagnation is detected, forces escalation or termination.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from enum import Enum
import logging

log = logging.getLogger("loop")


class ProgressSignal(Enum):
    """Types of progress signals per turn."""
    NEW_ASPECT_COVERED = "new_aspect_covered"
    CONFLICT_RESOLVED = "conflict_resolved"
    EVIDENCE_IMPROVED = "evidence_improved"
    COMMAND_REDUNDANT = "command_redundant"
    NO_PROGRESS = "no_progress"


@dataclass
class TurnSnapshot:
    """Snapshot of state at the end of each turn for progress tracking."""
    step: int
    covered_aspects: Set[str]  # aspects with coverage >= 0.7
    conflict_aspects: Set[str]  # aspects in disagree state
    max_confidence: float
    evidence_quality: float  # weighted average of evidence scores
    commands_issued: List[str]  # command names this turn
    agents_active: int
    tokens_used: int


@dataclass
class LoopProgressTracker:
    """Tracks decision loop progress and enforces forward momentum.

    Prevents infinite loops by:
      1. Detecting stagnation (no new aspects covered for N turns)
      2. Detecting redundant command patterns
      3. Forcing escalation when stuck (ESCALATE or STOP)

    Design principle: each iteration MUST either:
      - Cover a new aspect, OR
      - Resolve a conflict, OR
      - Improve evidence quality, OR
      - Escalate/STOP appropriately
    """

    def __init__(self,
                 stagnation_threshold: int = 3,
                 max_redundant_commands: int = 2):
        """Initialize tracker.

        Args:
            stagnation_threshold: Turns without new progress before forcing action
            max_redundant_commands: Consecutive redundant commands before flagging
        """
        self.stagnation_threshold = stagnation_threshold
        self.max_redundant_commands = max_redundant_commands
        self._history: List[TurnSnapshot] = []
        self._redundant_count: int = 0
        self._last_commands: Optional[Set[str]] = None

    def record_turn(self, snapshot: TurnSnapshot) -> ProgressSignal:
        """Record a turn and return the progress signal.

        Analyzes the turn to determine what kind of progress (if any) was made.
        Returns the dominant signal type for this turn.
        """
        if len(self._history) == 0:
            self._history.append(snapshot)
            return ProgressSignal.NEW_ASPECT_COVERED  # First turn always counts

        prev = self._history[-1]

        # Check for new aspects covered
        new_aspects = snapshot.covered_aspects - prev.covered_aspects
        has_new_aspect = len(new_aspects) > 0

        # Check for conflict resolution
        resolved_conflicts = prev.conflict_aspects - snapshot.conflict_aspects
        has_resolution = len(resolved_conflicts) > 0

        # Check for evidence quality improvement
        quality_improved = snapshot.evidence_quality > prev.evidence_quality + 0.05

        # Check for redundant commands
        is_redundant = (self._last_commands and
                       snapshot.commands_issued == list(self._last_commands))

        # Determine dominant signal
        if has_new_aspect:
            signal = ProgressSignal.NEW_ASPECT_COVERED
        elif has_resolution:
            signal = ProgressSignal.CONFLICT_RESOLVED
        elif quality_improved:
            signal = ProgressSignal.EVIDENCE_IMPROVED
        elif is_redundant:
            signal = ProgressSignal.COMMAND_REDUNDANT
        else:
            signal = ProgressSignal.NO_PROGRESS

        # Update redundant counter
        if signal == ProgressSignal.COMMAND_REDUNDANT:
            self._redundant_count += 1
        else:
            self._redundant_count = 0

        self._last_commands = set(snapshot.commands_issued)
        self._history.append(snapshot)

        return signal

    def is_stagnant(self) -> bool:
        """Check if the loop is stagnating (no progress for multiple turns)."""
        if len(self._history) < self.stagnation_threshold:
            return False

        recent = self._history[-self.stagnation_threshold:]

        # Check if any turn had meaningful progress
        for snap in recent:
            if hasattr(snap, '_signal') and snap._signal in (
                ProgressSignal.NEW_ASPECT_COVERED,
                ProgressSignal.CONFLICT_RESOLVED,
                ProgressSignal.EVIDENCE_IMPROVED
            ):
                return False

        # Check if evidence quality is trending up
        qualities = [s.evidence_quality for s in recent]
        if len(qualities) >= 2 and qualities[-1] > qualities[0] + 0.03:
            return False

        # Check for new aspects (even if not flagged as progress)
        recent_aspects = [s.covered_aspects for s in recent]
        if len(recent_aspects) >= 2:
            union_aspects = set()
            for aspects in recent_aspects:
                union_aspects.update(aspects)
                if len(union_aspects) > len(recent_aspects[0]):
                    return False

        return True

    def get_recommended_action(self) -> str:
        """Get recommended action when stagnation detected.

        Returns:
            "ESCALATE" - Force curated evidence inspection
            "STOP" - Terminate with best available answer
            "REPLAN" - Re-decompose the query
        """
        if len(self._history) < 2:
            return "WAIT"

        latest = self._history[-1]
        prev = self._history[-2]

        # If evidence quality is decent, STOP with current best
        if latest.evidence_quality > 0.7 and latest.max_confidence > 0.7:
            return "STOP"

        # If conflicts remain, ESCALATE to inspect evidence
        if latest.conflict_aspects:
            return "ESCALATE"

        # If coverage is low but max_steps is approaching, STOP
        if latest.step >= 35 and len(latest.covered_aspects) > 0:
            return "STOP"

        # Default: REPLAN to try different approach
        return "REPLAN"

    def should_force_stop(self, current_step: int, max_steps: int) -> bool:
        """Determine if we should force stop due to lack of progress.

        Args:
            current_step: Current step number
            max_steps: Maximum allowed steps

        Returns:
            True if we should force stop, False otherwise
        """
        # Force stop if approaching max_steps without progress
        if current_step >= max_steps - 2:
            if self.is_stagnant():
                log.info(f"[loop] Force stop at step {current_step}: "
                         f"stagnant + approaching max_steps")
                return True

        # Force stop if redundant commands for too long
        if self._redundant_count >= self.max_redundant_commands * 2:
            log.info(f"[loop] Force stop at step {current_step}: "
                     f"redundant commands ({self._redundant_count} consecutive)")
            return True

        return False

    def get_progress_summary(self) -> Dict[str, Any]:
        """Return a summary of loop progress for tracing/analysis.

        Includes:
        - Total turns
        - Aspects covered
        - Conflicts resolved
        - Evidence quality trend
        - Stagnation status
        """
        if not self._history:
            return {"status": "no_history"}

        latest = self._history[-1]
        first = self._history[0]

        aspects_covered = latest.covered_aspects
        aspects_resolved = first.conflict_aspects - latest.conflict_aspects

        quality_trend = "stable"
        if len(self._history) >= 3:
            recent_quality = [s.evidence_quality for s in self._history[-3:]]
            if recent_quality[-1] > recent_quality[0] + 0.05:
                quality_trend = "improving"
            elif recent_quality[-1] < recent_quality[0] - 0.05:
                quality_trend = "declining"

        return {
            "total_turns": len(self._history),
            "aspects_covered": list(aspects_covered),
            "n_aspects_covered": len(aspects_covered),
            "conflicts_resolved": list(aspects_resolved),
            "n_conflicts_resolved": len(aspects_resolved),
            "current_evidence_quality": latest.evidence_quality,
            "quality_trend": quality_trend,
            "is_stagnant": self.is_stagnant(),
            "redundant_command_count": self._redundant_count,
        }


def build_turn_snapshot(state: Any, pool: "EvidencePool",
                       commands_issued: List[str]) -> TurnSnapshot:
    """Build a TurnSnapshot from current state for progress tracking.

    Args:
        state: StateManager instance
        pool: EvidencePool instance
        commands_issued: Commands issued this turn

    Returns:
        TurnSnapshot with metrics for progress tracking
    """
    board = pool.status_board()

    # Extract covered aspects (coverage >= 0.7)
    covered_aspects = {
        row.get("aspect") for row in board
        if row.get("coverage", {}).get("covered", 0.0) >= 0.7
    }

    # Extract conflict aspects
    conflict_aspects = {
        agr.get("aspect") for agr in pool.aspect_agreements()
        if agr.get("agreement_state") == "disagree"
    }

    # Calculate evidence quality (weighted average of coverage and confidence)
    quality_scores = []
    for row in board:
        cov = row.get("coverage", {}).get("covered", 0.0)
        confs = [
            row.get("confidence", {}).get("retrieval_score", 0.0),
            row.get("confidence", {}).get("evidence_agreement", 0.0),
            row.get("confidence", {}).get("coverage", 0.0)
        ]
        max_conf = max((c for c in confs if c is not None), default=0.0)
        quality_scores.append(cov * 0.6 + max_conf * 0.4)

    evidence_quality = max(quality_scores) if quality_scores else 0.0
    max_confidence = pool.max_confidence()

    return TurnSnapshot(
        step=state.step_count,
        covered_aspects=covered_aspects,
        conflict_aspects=conflict_aspects,
        max_confidence=max_confidence,
        evidence_quality=evidence_quality,
        commands_issued=commands_issued,
        agents_active=len([a for a in board if a.get("status") not in ("killed", "failed")]),
        tokens_used=0  # TODO: integrate with budget tracking
    )


def calculate_redundancy_score(current_commands: List[str],
                                recent_history: List[TurnSnapshot]) -> float:
    """Calculate how redundant the current commands are.

    Higher score = more redundant (bad).
    Compares to recent command patterns.

    Returns 0.0 (not redundant) to 1.0 (completely redundant).
    """
    if not recent_history:
        return 0.0

    # Get command sets from recent turns
    recent_cmd_sets = [set(s.commands_issued) for s in recent_history[-3:]]
    current_set = set(current_commands)

    if not current_set:
        return 0.0

    # Calculate overlap with recent turns
    overlaps = []
    for cmd_set in recent_cmd_sets:
        if cmd_set:
            intersection = current_set & cmd_set
            union = current_set | cmd_set
            if union:
                overlaps.append(len(intersection) / len(union))

    return max(overlaps) if overlaps else 0.0
