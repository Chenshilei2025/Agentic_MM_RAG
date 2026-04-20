"""Deterministic runtime primitives used by the Orchestrator.

These are workflow-engine concerns, NOT reasoning:
  - TaskQueue        FIFO queue of pending sub-agent steps
  - WorkerPool       bounded parallel execution with timeout + retry
  - AgentLifecycle   per-agent state machine: PENDING / RUNNING / DONE / KILLED
"""
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from collections import deque
import time, uuid

PENDING, RUNNING, DONE, KILLED, FAILED = "PENDING", "RUNNING", "DONE", "KILLED", "FAILED"


@dataclass
class Subtask:
    """DA-produced decomposition of the user query. Each subtask binds an
    importance weight used by VoI gating hard rules."""
    id: str
    description: str
    aspect: str                        # short tag, e.g. "failure_handling"
    importance: float = 0.5            # [0,1] — hard rule 2 triggers at > 0.8
    modalities: List[str] = field(default_factory=list)
    # Round-3 (skills layer): CLIP-text embedding of the description,
    # populated by utils.subtask_embedder.embed_subtasks(). Used for
    # dedup (purpose A) and optionally retrieval-query alignment (purpose B).
    embedding: Optional[List[float]] = None


@dataclass
class Budget:
    """Soft token budget. Orchestrator counts used_tokens from provider usage
    reports. When used_tokens exceeds max_tokens, SKETCH/FULL authorization
    is refused (see voi_gating.budget_exceeded), but the run continues on
    Tier-2-only. Hard termination belongs to max_steps, not token budget.

    Round 6 — tier_breakdown tracks where tokens went (for paper analysis:
    how much of the budget is DA-prompt vs sub-agent-prompt vs skills vs
    tier-2 curated evidence). The orchestrator calls `add(category, n)`
    each time tokens are spent; used_tokens stays the authoritative sum.
    """
    max_tokens: int = 100_000
    used_tokens: int = 0
    # tier_breakdown keys: "da_prompt", "sub_agent", "skill_decompose",
    # "skill_reflect", "skill_replan", "curated_raw", "inspect", "other"
    tier_breakdown: Dict[str, int] = field(default_factory=dict)

    def remaining(self) -> int:
        return max(self.max_tokens - self.used_tokens, 0)

    def fraction_used(self) -> float:
        return min(self.used_tokens / max(self.max_tokens, 1), 1.0)

    def add(self, category: str, n_tokens: int) -> None:
        """Record n_tokens spent in `category`. Both used_tokens and the
        per-category bucket grow. Safe to call many times."""
        self.used_tokens += n_tokens
        self.tier_breakdown[category] = (
            self.tier_breakdown.get(category, 0) + n_tokens)


@dataclass
class AgentRecord:
    agent_id: str
    agent_type: str
    modality: str
    goal: str
    aspect: Optional[str] = None       # P2 — DA-assigned aspect tag
    status: str = PENDING
    stage: str = "intent"          # externalized from SubAgent; was per-instance
    retries_left: int = 2
    autonomy_level: str = "supervised"  # "supervised" | "autonomous"
    target_stage: str = "summary"       # autonomous runs until this stage done
    # Bounded history of recent tool names emitted by this sub-agent. Lets
    # stateless sub-agents see what they just did, so they advance the
    # Inspector pipeline instead of repeating retrieval.
    recent_actions: List[str] = field(default_factory=list)
    # Last retrieval result — injected into the sub-agent's next prompt
    # so it can pass actual candidates to rerank_text / rerank_visual /
    # dedupe_candidates instead of an empty list.
    last_retrieval: List[dict] = field(default_factory=list)
    last_reranked: List[dict] = field(default_factory=list)
    # Orchestrator-managed counter: how many times DA has issued
    # CONTINUE_RETRIEVAL on this agent. Used to cap infinite "retry".
    continue_round: int = 0
    created_at: float = field(default_factory=time.time)

@dataclass
class Task:
    task_id: str
    agent_id: str
    payload: dict                  # e.g. {"kind":"step"} — runtime signal only
    created_at: float = field(default_factory=time.time)

class TaskQueue:
    def __init__(self):
        self._q: deque[Task] = deque()

    def push(self, t: Task) -> None: self._q.append(t)
    def pop(self) -> Optional[Task]: return self._q.popleft() if self._q else None
    def drain(self) -> List[Task]:
        out, self._q = list(self._q), deque()
        return out
    def __len__(self): return len(self._q)

class WorkerPool:
    """Bounded parallel executor with timeout + retry. Deterministic: retries
    are counted per task, not per wallclock."""
    def __init__(self, max_workers: int = 4, default_timeout: float = 30.0):
        self._exec = ThreadPoolExecutor(max_workers=max_workers)
        self.timeout = default_timeout

    def run_many(self, fns: List[Callable[[], Any]], timeout: Optional[float] = None):
        """Submit fns in parallel; return list of (ok, result_or_exc)."""
        t = timeout or self.timeout
        futs = [self._exec.submit(f) for f in fns]
        out = []
        for fut in futs:
            try:
                out.append((True, fut.result(timeout=t)))
            except FutTimeout:
                out.append((False, TimeoutError(f"exceeded {t}s")))
            except Exception as e:
                out.append((False, e))
        return out

    def shutdown(self): self._exec.shutdown(wait=False)

def new_task_id() -> str: return "task-" + uuid.uuid4().hex[:8]
def new_agent_id() -> str: return "agent-" + uuid.uuid4().hex[:8]
