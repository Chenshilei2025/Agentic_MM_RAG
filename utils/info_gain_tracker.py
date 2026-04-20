"""Information-gain tracker — detects retrieval SATURATION.

Core idea: if over the last N decision turns neither max_confidence nor
mean_coverage has meaningfully grown, further retrieval is unlikely to
help. The DA should STOP with the best-partial answer rather than burn
more tokens. This is the paper's termination signal.

Why this matters for EMNLP:
  - Adaptive retrieval work (Self-RAG, FLARE, Adaptive-RAG) typically
    stops via confidence threshold alone, which misses the "stuck at
    medium confidence forever" failure mode.
  - Information-gain saturation is cheap, model-agnostic, and produces
    a calibrated trigger that's trivially ablatable.

Design:
  - Rolling window of last WINDOW_SIZE turn snapshots (default 3).
  - Each snapshot records (turn, max_conf, mean_coverage).
  - Saturation = (best aggregate score in window) - (worst) < DELTA.
  - `is_saturated` returns True only after window is full (no false
    positive in early turns).
  - Consumed by orchestrator.voi_gating.gate_request to DENY
    SPAWN/CONTINUE when saturated; STOP/INSPECT/READ_ARCHIVE still
    allowed so DA can finalize cleanly.
"""
from __future__ import annotations
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

# Defaults — picked conservatively; paper section will justify choices.
DEFAULT_WINDOW = 3         # need 3 turns of history before firing
DEFAULT_DELTA = 0.03       # score change below this = saturated


class InfoGainTracker:
    """Append-only history with a rolling saturation check.

    Thread-safety: single-threaded use inside the decision loop. The
    Orchestrator calls `record()` once per turn before dispatching a
    command, and `is_saturated()` when evaluating VoI for a new
    SPAWN / CONTINUE_RETRIEVAL request.
    """

    def __init__(self, window: int = DEFAULT_WINDOW,
                 delta: float = DEFAULT_DELTA):
        if window < 2:
            raise ValueError("window must be >= 2")
        self.window = window
        self.delta = delta
        self._history: Deque[Dict[str, float]] = deque(maxlen=window)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------
    def record(self, turn: int, max_conf: float,
               mean_coverage: float) -> None:
        """Call once per DA turn AFTER new evidence has been written
        but BEFORE the next command is dispatched."""
        self._history.append({
            "turn": int(turn),
            "max_conf": float(max_conf),
            "mean_coverage": float(mean_coverage),
            # Aggregate score used by saturation check — equal weight.
            # Paper section: also try max over axes / coverage-only variants.
            "score": 0.5 * float(max_conf) + 0.5 * float(mean_coverage),
        })

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def is_saturated(self) -> bool:
        """True when full window is collected AND score spread < delta."""
        if len(self._history) < self.window:
            return False
        scores = [h["score"] for h in self._history]
        return (max(scores) - min(scores)) < self.delta

    def growth_last_n(self, n: Optional[int] = None) -> float:
        """Return score growth (latest − earliest) over last n entries.
        n=None means over the full window. Returns 0.0 if not enough data."""
        n = n or self.window
        if len(self._history) < n:
            return 0.0
        hs = list(self._history)[-n:]
        return hs[-1]["score"] - hs[0]["score"]

    def snapshot(self) -> Dict[str, object]:
        """Serializable state for trace/debug. No internal details exposed."""
        return {
            "window": self.window,
            "delta": self.delta,
            "n_recorded": len(self._history),
            "history": list(self._history),
            "saturated": self.is_saturated(),
            "growth": self.growth_last_n(),
        }

    def reset(self) -> None:
        self._history.clear()
