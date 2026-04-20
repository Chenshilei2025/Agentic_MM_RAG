"""Global Tool Registry. Tools register themselves; Orchestrator looks them up.
Tools are ATOMIC OPERATIONS ONLY — no reasoning, no LLM calls."""
from typing import Callable, Dict, Any
from dataclasses import dataclass

@dataclass
class ToolSpec:
    name: str
    description: str
    schema: Dict[str, Any]   # JSON schema for arguments
    handler: Callable[..., Any]

class ToolRegistry:
    _tools: Dict[str, ToolSpec] = {}

    @classmethod
    def register(cls, spec: ToolSpec) -> None:
        if spec.name in cls._tools:
            raise ValueError(f"Tool already registered: {spec.name}")
        cls._tools[spec.name] = spec

    @classmethod
    def get(cls, name: str) -> ToolSpec:
        if name not in cls._tools:
            raise KeyError(f"Unknown tool: {name}")
        return cls._tools[name]

    @classmethod
    def list_specs(cls) -> Dict[str, ToolSpec]:
        return dict(cls._tools)

def tool(name: str, description: str, schema: Dict[str, Any]):
    """Decorator to register a tool handler."""
    def deco(fn: Callable[..., Any]):
        ToolRegistry.register(ToolSpec(name=name, description=description,
                                       schema=schema, handler=fn))
        return fn
    return deco
