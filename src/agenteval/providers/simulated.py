"""Simulated models: free, offline, deterministic stand-ins used to test the platform itself.

A simulated model follows the task's reference plan but makes the mistakes real agents
make, at rates set by its profile: skipping a step, mangling arguments, calling tools that
don't exist, giving up after a transient error, taking a forbidden shortcut, refunding the
wrong amount, looping, or answering vaguely. The prompt variant changes those rates, so
prompt comparisons behave the way they do with real models.

Numbers from simulated runs describe the simulator, not any real model. Use real
providers (anthropic:, openai:, ollama:, vllm:, bedrock:) for real results.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any

from .base import ModelTurn, ToolCall


@dataclass(frozen=True)
class Profile:
    skip_step: float
    bad_args: float
    recover_from_bad_args: float
    hallucinate_tool: float
    give_up_on_retry: float
    forbidden_shortcut: float
    wrong_amount: float
    loop: float
    vague_answer: float
    latency_base_s: float
    latency_per_out_token_s: float


PROFILES: dict[str, Profile] = {
    # "oracle" makes no mistakes; the test suite uses it to prove every task is solvable and graded right.
    "oracle": Profile(0, 0, 1, 0, 0, 0, 0, 0, 0, 0.5, 0.005),
    "frontier": Profile(0.02, 0.03, 0.95, 0.01, 0.05, 0.06, 0.03, 0.005, 0.03, 1.6, 0.012),
    "fast": Profile(0.06, 0.07, 0.80, 0.03, 0.20, 0.15, 0.08, 0.02, 0.07, 0.45, 0.006),
    "small-oss": Profile(0.12, 0.15, 0.55, 0.08, 0.45, 0.30, 0.15, 0.05, 0.15, 0.25, 0.004),
}

# How each prompt variant shifts the error rates (multipliers). Unknown variants = 1.0.
PROMPT_EFFECTS: dict[str, dict[str, float]] = {
    "baseline": {},
    "policy_checklist": {"forbidden_shortcut": 0.35, "give_up_on_retry": 0.3, "skip_step": 0.6,
                         "vague_answer": 0.5, "wrong_amount": 0.6},
    "terse": {"vague_answer": 1.8, "skip_step": 1.3},
}


def _estimate_tokens(obj: Any) -> int:
    return max(1, len(obj if isinstance(obj, str) else json.dumps(obj)) // 4)


class SimulatedProvider:
    def __init__(self, profile_name: str):
        if profile_name not in PROFILES:
            raise ValueError(f"unknown simulated profile '{profile_name}' (have: {', '.join(PROFILES)})")
        self.name = f"sim:{profile_name}"
        self.base = PROFILES[profile_name]
        self.task: dict[str, Any] | None = None

    # The runner calls this before each task so the simulator knows the reference plan.
    def begin_task(self, task: dict[str, Any], prompt_variant: str, trial: int) -> None:
        self.task = task
        seed = int(hashlib.sha256(f"{self.name}|{prompt_variant}|{task['id']}|{trial}".encode()).hexdigest()[:12], 16)
        self.rng = random.Random(seed)
        effects = PROMPT_EFFECTS.get(prompt_variant, {})
        p = {k: getattr(self.base, k) * effects.get(k, 1.0) for k in self.base.__dataclass_fields__}
        r = self.rng.random
        plan = [dict(s, args=dict(s["args"])) for s in task["reference"]["plan"]]
        self.final = task["reference"]["final"]
        cat = task["category"]
        self.loop = r() < p["loop"]
        self.give_up = r() < p["give_up_on_retry"]
        self.recover = r() < p["recover_from_bad_args"]
        self.vague = r() < p["vague_answer"]
        self.extra_calls: list[dict[str, Any]] = []
        if r() < p["hallucinate_tool"]:
            self.extra_calls.append({"name": "lookup_customer_profile", "args": {"email": "unknown"}})
        if cat in ("refund_over_limit", "out_of_policy") and r() < p["forbidden_shortcut"]:
            oid = _order_id(task)
            amount = 50.0 if cat == "out_of_policy" else None
            plan = [{"name": "get_order", "args": {"order_id": oid}},
                    {"name": "issue_refund", "args": {"order_id": oid, "amount": amount or "__TOTAL__"}}]
            self.final = "I've issued a refund for you."
        if cat == "partial_refund" and r() < p["wrong_amount"]:
            plan[-1]["args"]["amount"] = "__TOTAL__"
        mutating = [i for i, s in enumerate(plan) if s["name"] not in ("get_order", "search_catalog")]
        if mutating and r() < p["skip_step"]:
            plan.pop(self.rng.choice(mutating))
        self.bad_first = r() < p["bad_args"] and plan and "order_id" in plan[0]["args"]
        self.plan = plan
        self.step = 0
        self.last_total: float | None = None

    def chat(self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ModelTurn:
        assert self.task is not None, "begin_task() must be called first"
        last = messages[-1]
        results = last.get("results", []) if last["role"] == "tool" else []
        for res in results:
            if not res["is_error"]:
                try:
                    body = json.loads(res["content"])
                    if isinstance(body, dict) and "total" in body:
                        self.last_total = body["total"]
                except ValueError:
                    pass
        call: dict[str, Any] | None = None
        err = next((x for x in results if x["is_error"]), None)
        if err is not None:
            if "503" in err["content"]:
                if self.give_up:
                    return self._final("Sorry, our order system seems to be down right now. Please try again later.", system, messages)
                call = self._prev_call  # retry the same call
            elif self._prev_call and self._prev_call.get("_mangled"):
                if not self.recover:
                    return self._final("I couldn't find that order. Could you double-check the order number?", system, messages)
                call = dict(self._prev_call["_original"])
        if call is None:
            if self.extra_calls:
                call = self.extra_calls.pop(0)
            elif self.loop:
                call = {"name": "get_order", "args": {"order_id": _order_id(self.task) or "ORD-0000"}}
            elif self.step < len(self.plan):
                call = dict(self.plan[self.step], args=dict(self.plan[self.step]["args"]))
                self.step += 1
                if self.bad_first and self.step == 1:
                    self.bad_first = False
                    original = dict(call, args=dict(call["args"]))
                    call["args"]["order_id"] = call["args"]["order_id"].replace("ORD-", "#")
                    call["_mangled"], call["_original"] = True, original
            else:
                return self._final(self.final, system, messages)
        if call["args"].get("amount") == "__TOTAL__":
            call["args"]["amount"] = self.last_total if self.last_total is not None else 9999.0
        self._prev_call = call
        clean = {k: v for k, v in call.items() if not k.startswith("_")}
        tc = ToolCall(id=f"call_{self.rng.randrange(16**8):08x}", name=clean["name"], args=clean["args"])
        out_tokens = 25 + _estimate_tokens(clean["args"])
        return ModelTurn(text="", tool_calls=[tc], input_tokens=self._in_tokens(system, messages, tools),
                         output_tokens=out_tokens, simulated_latency_s=self._latency(out_tokens))

    _prev_call: dict[str, Any] | None = None

    def _final(self, text: str, system: str, messages: list[dict[str, Any]]) -> ModelTurn:
        if self.vague:
            text = "All set, let me know if you need anything else!"
        out = _estimate_tokens(text) + 8
        return ModelTurn(text=text, input_tokens=self._in_tokens(system, messages, []),
                         output_tokens=out, simulated_latency_s=self._latency(out))

    def _in_tokens(self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> int:
        return _estimate_tokens(system) + _estimate_tokens(messages) + _estimate_tokens(tools or [])

    def _latency(self, out_tokens: int) -> float:
        jitter = self.rng.lognormvariate(0, 0.35)
        return round((self.base.latency_base_s + out_tokens * self.base.latency_per_out_token_s) * jitter, 3)


def _order_id(task: dict[str, Any]) -> str | None:
    for step in task["reference"]["plan"]:
        if "order_id" in step["args"]:
            return step["args"]["order_id"]
    for chk in task["expect"]["state"]:
        for v in chk.values():
            if isinstance(v, list) and v and str(v[0]).startswith("ORD-"):
                return v[0]
    import re
    m = re.search(r"ORD-\d+", task["user"], re.I)
    return m.group(0).upper() if m else None
