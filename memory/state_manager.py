"""System state. Holds agents, feedback queues, final-answer slot, trace.
Access to spawn/feedback queues is thread-safe under a single lock."""
import threading
from typing import Dict, List, Optional, Any

class StateManager:
    def __init__(self, query: str):
        self.query = query
        self.agents: Dict[str, Any] = {}
        self.active: Dict[str, bool] = {}
        self.feedback: Dict[str, List[str]] = {}
        self.trace: List[dict] = []
        self.final: Optional[dict] = None
        self.step_count: int = 0
        self.recent_spawns: List[str] = []
        self.pending_inspect: List[Dict[str, Any]] = []
        self.inspect_count: int = 0
        # READ_ARCHIVE drains tier-3 archive content into here for one-time
        # injection into the DA's NEXT prompt. Auto-consumed each turn.
        # List of {"agent_id": str, "content": str, "sources": list}
        self.pending_archive: List[Dict[str, Any]] = []
        # Subtask decomposition — populated once at run start by DA, if any.
        # Each entry is a runtime.Subtask. DA and VoI gating both consult this.
        self.subtasks: List[Any] = []
        # Token budget — enforced softly by VoI gating, see voi_gating.py
        self.budget: Any = None     # runtime.Budget, populated by Orchestrator
        # VoI decisions trace — populated each time VoI gating is consulted,
        # for ablation analysis (e.g. "how often would VoI block a request?")
        self.voi_decisions: List[Dict[str, Any]] = []
        # P2 round-1 guardrails:
        # - aborted_agents: set of agent_ids that have already been ABORTed
        #   once. Second ABORT on the same agent is rejected (prevents
        #   DA from looping abort/respawn on the same problem).
        # - revise_trace: every REVISE_SUBTASK is logged here for ablation
        #   ("what would happen if we disallowed subtask revision?")
        self.aborted_agents: set = set()
        self.revise_trace: List[Dict[str, Any]] = []
        # Round 2 — reflect/replan skill integration
        # pending_reflect_verdict: verdict dict from a completed REFLECT call
        # that should be surfaced at the top of DA's next prompt. Consumed
        # on read (one-shot, like pending_inspect).
        self.pending_reflect_verdict: Optional[Dict[str, Any]] = None
        # reflect_verdicts: append-only history, for trace/ablation export
        self.reflect_verdicts: List[Dict[str, Any]] = []
        # replan_traces: each REPLAN call records its patch + what was
        # actually dispatched, so we can measure "replan efficacy" later
        self.replan_traces: List[Dict[str, Any]] = []
        # Round 4 — info-gain saturation tracker. Recorded each DA turn.
        from utils.info_gain_tracker import InfoGainTracker
        self.info_gain_tracker = InfoGainTracker()
        # Round 4 — loop progress tracker. Detects stagnation and prevents
        # infinite loops. Records TurnSnapshot at end of each iteration.
        from utils.loop_progress import LoopProgressTracker
        self.loop_progress_tracker = LoopProgressTracker(
            stagnation_threshold=3,
            max_redundant_commands=2
        )
        # Round 4 — RESOLVE_CONFLICT decisions, keyed by aspect.
        # Used at answer synthesis time: `trust_agent_id` wins over the
        # rest of that aspect's findings.
        self.conflict_resolutions: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def register_agent(self, agent) -> None:
        with self._lock:
            self.agents[agent.id] = agent
            self.active[agent.id] = True
            self.feedback.setdefault(agent.id, [])

    def deactivate_agent(self, agent_id: str) -> None:
        with self._lock:
            self.active[agent_id] = False

    def active_agents(self) -> list:
        with self._lock:
            return [a for aid, a in self.agents.items()
                    if self.active.get(aid)]

    def push_feedback(self, agent_id: str, msg: str) -> None:
        with self._lock:
            self.feedback.setdefault(agent_id, []).append(msg)

    def drain_feedback(self, agent_id: str) -> List[str]:
        with self._lock:
            msgs = self.feedback.get(agent_id, [])
            self.feedback[agent_id] = []
            return msgs

    def record_spawn(self, agent_id: str) -> None:
        """Called by Orchestrator when spawning. Thread-safe append."""
        with self._lock:
            self.recent_spawns.append(agent_id)

    def drain_recent_spawns(self) -> List[str]:
        """Atomic read-and-clear. Called by Decision Agent exactly once
        per turn before building its prompt."""
        with self._lock:
            out = list(self.recent_spawns)
            self.recent_spawns = []
            return out

    def finalize(self, answer: str, confidence: float,
                 reason: str = "completed") -> None:
        with self._lock:
            self.final = {"answer": answer, "confidence": confidence,
                          "reason": reason}

    def is_done(self) -> bool:
        return self.final is not None

    def record(self, entry: dict) -> None:
        with self._lock:
            self.trace.append(entry)
