"""Decision Agent skills — modular prompt + provider-call units.

Each skill is a focused, single-responsibility capability the DA can
invoke independently: decompose / reflect / replan. Isolating them:
  - keeps each prompt small and optimisable
  - makes per-skill trace/eval possible (consumers can measure decompose
    quality separately from reflect quality)
  - lets us swap a skill's prompt or provider without touching the others

Each skill module exposes a single `run(...)` function that takes the
minimal inputs it needs and returns a structured result (dict / list).
Failures are caught locally and returned as None or empty, never thrown,
so the core decision loop is robust to any one skill having a bad turn.
"""
