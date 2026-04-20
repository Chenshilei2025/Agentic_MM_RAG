# Agentic Multimodal RAG

Production-grade, deterministic, tool-driven multi-agent orchestration system.
Built from first principles. No LangChain, no AutoGPT, no RL.

## Quickstart

```bash
pip install pyyaml anthropic
python -m tests.run_tests                               # 10 tests
python main.py --query "who executes tools?" --provider scripted
# With a real API key:
export ANTHROPIC_API_KEY=sk-...
python main.py --query "..." --provider anthropic
```

## Architecture

```
Decision Agent → action (JSON only)
       ↓
Orchestrator  → executes tool (ONLY executor)
       ↓
Tool          → atomic operation
       ↓
State + Evidence Pool updated
       ↓
Loop continues until stopping rule
```

### Hard-constraint compliance

| Constraint | Enforced in |
|---|---|
| Orchestrator is sole executor | `orchestrator/controller.py::_execute` |
| Agents are pure policy | `agents/base.py` — no `call_tool`, no state writes |
| Provider is the only LLM path | `providers/base_provider.py` |
| Everything is a tool | `tools/builtin.py` — all 7 required tools |
| Structured JSON output only | `cli/schemas/action.py::parse + validate` |
| No hardcoded prompts | `prompts/prompt_builder.py` is the only source |
| Progressive disclosure | `SubAgent.STAGES = (intent, summary, full)` |
| Deterministic stopping | `Orchestrator._should_stop` — `conf > 0.9 AND coverage >= 0.8` |
| Registry patterns | `ToolRegistry`, `ProviderRegistry`, `SubAgentFactory` |
| Observability | `utils/logger.py` + `state.trace` records every call |

### Project Layout
```
agents/       base.py decision_agent.py sub_agent.py factory.py
tools/        registry.py builtin.py  (+ retrieval/ agent_control/ evidence/ communication/)
providers/    base_provider.py anthropic_provider.py mock_provider.py registry.py
cli/          parser.py validator.py schemas/action.py
memory/       evidence_pool.py state_manager.py store.py
orchestrator/ controller.py
prompts/      prompt_builder.py
config/       system.yaml
utils/        logger.py
tests/        run_tests.py
main.py
```

## Extension points

- **Swap the vector store**: replace `memory/store.InMemoryStore` with any object exposing `.search(query, k) -> list[dict]`.
- **Add a provider**: subclass `BaseProvider`, register with `ProviderRegistry.register(...)`.
- **Add a tool**: decorate a function with `@tool(name, description, schema)` in any module imported at startup.
- **Whitelist tools per agent**: `config/system.yaml` → `agents.decision.allowed_tools` / `agents.subagent.allowed_tools`.

## What's deliberately out of scope

- No RL, no reward models, no training loops (forbidden by spec)
- No agent-side tool execution path (enforced architecturally)
- No free-form agent output (parser + validator reject it)
