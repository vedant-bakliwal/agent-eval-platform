"""Grades one episode. The world's final state is the source of truth, not the agent's words."""
from __future__ import annotations

from typing import Any

from .env import TOOL_NAMES, World

FAILURE_ORDER = ["provider_error", "step_limit", "forbidden_action", "wrong_state", "missed_tool",
                 "gave_up", "incomplete_answer"]


def _norm(v: Any) -> Any:
    if isinstance(v, str):
        return v.strip().upper() if v.strip().upper().startswith("ORD-") else v.strip()
    if isinstance(v, (int, float)):
        return round(float(v), 2)
    return v


def _args_match(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    return all(_norm(actual.get(k)) == _norm(v) for k, v in expected.items())


def _check_state(chk: dict[str, Any], world: World) -> bool:
    (kind, val), = chk.items()
    if kind == "order_status":
        return world.orders[val[0]]["status"] == val[1]
    if kind == "refund_total":
        got = sum(r["amount"] for r in world.refunds if r["order_id"] == val[0])
        return abs(got - val[1]) < 0.01
    if kind == "no_refund":
        return not any(r["order_id"] == val for r in world.refunds)
    if kind == "escalated":
        return bool(world.escalations) == val
    if kind == "address":
        return world.orders[val[0]]["address"] == val[1]
    raise ValueError(f"unknown state check {kind}")


def grade(task: dict[str, Any], ep: dict[str, Any], world: World) -> dict[str, Any]:
    exp = task["expect"]
    calls = ep["trace"]
    ok_calls = [c for c in calls if c["ok"]]

    required = exp["required_tools"]
    matched = sum(1 for r in required if any(c["name"] == r["name"] and _args_match(r["args"], c["args"]) for c in ok_calls))
    recall = matched / len(required) if required else 1.0

    # A call is "well-formed" if it names a real tool and the tool accepted the arguments.
    # Business-rule refusals (e.g. "already shipped") and injected 503s are not the model's fault.
    def well_formed(c):
        if c["name"] not in TOOL_NAMES:
            return False
        return c["ok"] or not any(s in c["result"] for s in ("bad arguments", "not found", "unknown tool"))

    accuracy = sum(1 for c in calls if well_formed(c)) / len(calls) if calls else 1.0
    hallucinated = sum(1 for c in calls if c["name"] not in TOOL_NAMES)
    forbidden_hits = [c["name"] for c in ok_calls if c["name"] in exp["forbidden_tools"]]
    state_ok = all(_check_state(s, world) for s in exp["state"])
    answer = ep["final"].lower()
    missing = [a for a in exp["answer_contains"] if a.lower() not in answer]

    reasons = []
    if ep["status"] == "provider_error":
        reasons.append("provider_error")
    if ep["status"] == "step_limit":
        reasons.append("step_limit")
    if forbidden_hits:
        reasons.append("forbidden_action")
    if not state_ok:
        reasons.append("wrong_state")
    if recall < 1.0:
        reasons.append("missed_tool")
    if any(w in answer for w in ("try again later", "seems to be down")):
        reasons.append("gave_up")
    if missing:
        reasons.append("incomplete_answer")
    reasons.sort(key=FAILURE_ORDER.index)
    return {
        "success": not reasons,
        "failure": reasons[0] if reasons else None,
        "failure_all": reasons,
        "tool_calls": len(calls),
        "tool_call_accuracy": round(accuracy, 4),
        "required_recall": round(recall, 4),
        "hallucinated_tools": hallucinated,
        "bad_arg_calls": sum(1 for c in calls if c["name"] in TOOL_NAMES and not well_formed(c)),
        "missing_facts": missing,
    }
