"""Action schema: {"tool_name": str, "arguments": dict}. No free-form output."""
import json
from typing import Any, Dict
from tools.registry import ToolRegistry

ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "tool_name": {"type": "string"},
        "arguments": {"type": "object"},
    },
    "required": ["tool_name", "arguments"],
    "additionalProperties": False,
}

class ParseError(Exception): ...
class ValidationError(Exception): ...


def _iter_balanced_objects(s: str):
    """Yield every substring of s that is a balanced {...} block.
    Skips characters inside string literals. Multiple independent blocks
    at top level are all yielded in order."""
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(s):
        if in_str:
            if esc:         esc = False
            elif ch == "\\": esc = True
            elif ch == '"': in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            if depth == 0: start = i
            depth += 1
        elif ch == "}":
            if depth == 0: continue
            depth -= 1
            if depth == 0 and start >= 0:
                yield s[start:i + 1]
                start = -1


def _extract_first_json_object(s: str) -> str:
    """Backward-compatible single-object extractor."""
    for cand in _iter_balanced_objects(s):
        return cand
    raise ParseError("no balanced JSON object found")


def _strip_fences(s: str) -> str:
    """Strip markdown code fences anywhere in the string, not just at start.
    Reasoner models sometimes wrap JSON in ```json ... ``` blocks."""
    import re
    # Remove leading/trailing whitespace.
    s = s.strip()
    # If the response is entirely a fenced block, take its body.
    if s.startswith("```"):
        parts = s.split("\n", 1)
        body = parts[1] if len(parts) > 1 else ""
        if "```" in body:
            body = body[:body.rfind("```")]
        s = body.strip()
    # Also strip inline ```json ... ``` fences that appear mid-response.
    s = re.sub(r"```(?:json)?\s*", "", s)
    s = s.replace("```", "")
    return s


def parse(raw: str) -> Dict[str, Any]:
    """Extract and parse a JSON command/action from LLM output.
    Tolerant of: prose before/after JSON, markdown fences, multiple
    candidate blocks (picks first that contains tool_name or command)."""
    if not isinstance(raw, str):
        raise ParseError("agent output must be a string")
    stripped = _strip_fences(raw)
    # Try every balanced {...} block; return the first that looks like an
    # action (tool_name) or command (command).
    last_err = None
    for cand in _iter_balanced_objects(stripped):
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError as e:
            last_err = e
            continue
        if not isinstance(obj, dict):
            continue
        # Filter out obviously-wrong candidates (reasoner sometimes writes
        # little `{key: value}` fragments in its preamble).
        if "tool_name" in obj or "command" in obj:
            return obj
    if last_err:
        raise ParseError(f"no valid action/command object in output: {last_err}")
    raise ParseError("no balanced JSON object with tool_name/command found")


def validate(action: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(action, dict):
        raise ValidationError("action must be an object")
    for k in ("tool_name", "arguments"):
        if k not in action:
            raise ValidationError(f"missing key: {k}")
    if not isinstance(action["tool_name"], str):
        raise ValidationError("tool_name must be string")
    if not isinstance(action["arguments"], dict):
        raise ValidationError("arguments must be object")
    ToolRegistry.get(action["tool_name"])
    return action
