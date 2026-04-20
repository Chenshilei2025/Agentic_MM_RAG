"""Control command layer — Decision Agent's vocabulary.

Commands are POLICY intentions. The Orchestrator translates commands into
internal tool dispatches. This keeps the Decision Agent high-level and
the Orchestrator deterministic.
"""
from typing import Any, Dict

MODALITIES = ("doc_text", "doc_visual", "video_text", "video_visual")

COMMAND_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "SPAWN_AGENT": {
        "required": ["agent_type", "modality", "goal"],
        "properties": {"agent_type": "string", "modality": "enum:modality",
                       "goal": "string", "top_k": "int?",
                       "aspect": "string?",
                       "autonomy_level": "enum:supervised|autonomous"},
    },
    "SPAWN_AGENTS": {
        "required": ["specs"],
        "properties": {"specs": "list[spawn_spec]"},
    },
    "KILL_AGENT": {
        "required": ["agent_id"],
        "properties": {"agent_id": "string", "reason": "string?"},
    },
    "SWITCH_MODALITY": {
        "required": ["agent_id", "modality"],
        "properties": {"agent_id": "string", "modality": "enum:modality"},
    },
    "CONTINUE_RETRIEVAL": {
        "required": ["agent_id"],
        "properties": {"agent_id": "string", "hint": "string?"},
    },
    "REQUEST_CURATED_EVIDENCE": {
        # Tier-2 disclosure — DA asks a sub-agent (or a specific set of ids)
        # to surface PRE-FILTERED evidence it used to form its Tier-1 finding.
        # Unifies the old SKETCH (curated list of candidates with relevance)
        # + INSPECT (raw asset blobs). Two modes:
        #   - agent_id only            → agent-level: top-K candidates the
        #                                sub-agent already reranked
        #   - agent_id + ids=[...]     → id-level: only those ids, curated
        # `with_raw=true` attaches the original content/base64 blobs for each
        # candidate (~5-10x cost; only use for conflict resolution or
        # low-confidence high-importance subtasks).
        "required": ["agent_id"],
        "properties": {"agent_id": "string",
                       "ids": "list[string]?",
                       "with_raw": "boolean?",
                       "reason": "string?"},
    },
    "REFLECT": {
        # DA explicitly invokes the reflect_skill and blocks on its verdict
        # before issuing the next turn's command. Use when DA is uncertain
        # whether it should ANSWER, ESCALATE, REPLAN, or WAIT. The verdict
        # gets injected into the DA's next prompt top.
        "required": [],
        "properties": {"focus_aspect": "string?",
                       "reason": "string?"},
    },
    "REPLAN": {
        # DA explicitly invokes the replan_skill to produce a minimal patch
        # (extend / revise / abort_respawn) based on the current state +
        # an (optional) REFLECT verdict. Patch is executed through existing
        # EXTEND_SUBTASKS / REVISE_SUBTASK / ABORT_AND_RESPAWN dispatch —
        # all guardrails apply (cap 10 subtasks, 1 abort per agent).
        "required": [],
        "properties": {"reason": "string?"},
    },
    "INSPECT_EVIDENCE": {
        # (Retained for back-compat during the transition. New code should
        # prefer REQUEST_CURATED_EVIDENCE(with_raw=True, ids=[...]).)
        "required": ["ids"],
        "properties": {"ids": "list[string]",
                       "reason": "string?"},
    },
    "EXTEND_SUBTASKS": {
        "required": ["subtasks"],
        "properties": {"subtasks": "list[subtask_spec]"},
    },
    "REVISE_SUBTASK": {
        "required": ["id"],
        "properties": {"id": "string",
                       "modalities": "list[string]?",
                       "importance": "number?",
                       "description": "string?"},
    },
    "ABORT_AND_RESPAWN": {
        "required": ["agent_id"],
        "properties": {"agent_id": "string",
                       "new_goal": "string?",
                       "new_modality": "enum:modality?",
                       "new_aspect": "string?",
                       "reason": "string?"},
    },
    "RESOLVE_CONFLICT": {
        # Round 4 — DA explicitly arbitrates a cross-modal conflict on an
        # aspect. When aspect_agreements.agreement_state=="disagree" and
        # further retrieval won't help, DA issues RESOLVE_CONFLICT to
        # declare which side to trust (or that the conflict is fundamental
        # and the answer must reflect it). The resolution is consulted
        # at answer synthesis: trust_agent's finding overrides others for
        # that aspect.
        "required": ["aspect", "resolution"],
        "properties": {
            "aspect":          "string",
            "resolution":      "string",   # trust_one | unresolvable | complementary
            "trust_agent_id":  "string?",  # required when resolution=trust_one
            "reason":          "string?",
        },
    },
    "STOP_TASK": {
        "required": ["reason"],
        "properties": {"reason": "string", "answer": "string?",
                       "confidence": "number?"},
    },
}

class CommandError(Exception): ...

def _validate_spawn_args(args: Dict[str, Any]) -> None:
    for k in ("agent_type", "modality", "goal"):
        if k not in args:
            raise CommandError(f"spawn spec missing: {k}")
    if args["modality"] not in MODALITIES:
        raise CommandError(f"invalid modality: {args['modality']}")

def validate_command(cmd: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(cmd, dict):
        raise CommandError("command must be an object")
    if "command" not in cmd or "arguments" not in cmd:
        raise CommandError("missing 'command' or 'arguments'")
    name = cmd["command"]
    if name not in COMMAND_SCHEMAS:
        raise CommandError(f"unknown command: {name}")
    args = cmd["arguments"]
    if not isinstance(args, dict):
        raise CommandError("arguments must be object")
    for k in COMMAND_SCHEMAS[name]["required"]:
        if k not in args:
            raise CommandError(f"{name} missing required arg: {k}")

    # Per-command argument validation
    if name == "SPAWN_AGENT":
        _validate_spawn_args(args)
    elif name == "SPAWN_AGENTS":
        specs = args["specs"]
        if not isinstance(specs, list) or not specs:
            raise CommandError("SPAWN_AGENTS.specs must be non-empty list")
        if len(specs) > 8:
            raise CommandError("SPAWN_AGENTS.specs max 8 per turn")
        for s in specs:
            if not isinstance(s, dict):
                raise CommandError("each spawn spec must be an object")
            _validate_spawn_args(s)
    elif "modality" in args and args["modality"] not in MODALITIES:
        raise CommandError(f"invalid modality: {args['modality']}")
    return cmd
