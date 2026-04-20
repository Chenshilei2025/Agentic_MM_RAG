"""Mock provider that emits scripted JSON actions. For testing without an API key."""
from typing import List, Dict, Callable
from providers.base_provider import BaseProvider

class ScriptedProvider(BaseProvider):
    """Emits the next string from a script on each `complete` call.

    Script may contain strings OR callables(messages, system) -> str, so
    tests can make decisions conditional on the prompt.
    """
    name = "scripted"

    def __init__(self, script: List):
        self._script = list(script)
        self._i = 0

    def complete(self, messages: List[Dict[str, str]], system: str = "",
                 **kw) -> str:
        if self._i >= len(self._script):
            # Default fallback: emit a STOP_TASK command so the orchestrator
            # terminates cleanly instead of exhausting max_steps.
            return ('{"command":"STOP_TASK","arguments":'
                    '{"reason":"script_exhausted","answer":"",'
                    '"confidence":0.0}}')
        item = self._script[self._i]
        self._i += 1
        return item(messages, system) if callable(item) else item
