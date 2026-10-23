"""Runs the agent loop for every (model, prompt, task, trial) and records a full trace."""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from .env import TOOL_SPECS, ToolError, World
from .grading import grade
from .providers.base import ProviderError, price_for
from .providers.real import make_provider


def run_episode(provider, world: World, task: dict[str, Any], system: str) -> dict[str, Any]:
    world.flaky = dict(task["setup"].get("flaky", {}))
    messages: list[dict[str, Any]] = [{"role": "user", "content": task["user"]}]
    trace: list[dict[str, Any]] = []
    tokens_in = tokens_out = 0
    latency = 0.0
    final_text, status, error = "", "ok", None
    max_steps = task["expect"]["max_steps"]
    for _ in range(max_steps):
        t0 = time.perf_counter()
        try:
            turn = provider.chat(system, messages, TOOL_SPECS)
        except ProviderError as exc:
            status, error = "provider_error", str(exc)
            break
        latency += turn.simulated_latency_s if turn.simulated_latency_s is not None else time.perf_counter() - t0
        tokens_in += turn.input_tokens
        tokens_out += turn.output_tokens
        messages.append({"role": "assistant", "text": turn.text,
                         "tool_calls": [{"id": c.id, "name": c.name, "args": c.args} for c in turn.tool_calls]})
        if not turn.tool_calls:
            final_text = turn.text
            break
        results = []
        for c in turn.tool_calls:
            try:
                out = world.call(c.name, c.args)
                content, is_error = json.dumps(out), False
            except ToolError as exc:
                content, is_error = f"ERROR: {exc}", True
            trace.append({"name": c.name, "args": c.args, "ok": not is_error, "result": content[:500]})
            results.append({"id": c.id, "name": c.name, "content": content, "is_error": is_error})
        messages.append({"role": "tool", "results": results})
    else:
        status = "step_limit"
    return {"final": final_text, "trace": trace, "status": status, "error": error,
            "tokens_in": tokens_in, "tokens_out": tokens_out, "latency_s": round(latency, 3),
            "turns": sum(1 for m in messages if m["role"] == "assistant")}


def run(suite: tuple[World, list[dict[str, Any]]], models: list[str], prompts: dict[str, str], trials: int,
        out_path: Path, workers: int = 8, prices: dict[str, tuple[float, float]] | None = None,
        limit: int | None = None, on_progress: Callable[[int, int], None] | None = None) -> list[dict[str, Any]]:
    world, tasks = suite
    tasks = tasks[:limit] if limit else tasks
    jobs = [(m, p, t, k) for m in models for p in prompts for t in tasks for k in range(trials)]
    prices = prices or {}

    def one(job):
        model, prompt_name, task, trial = job
        provider = make_provider(model)  # one instance per job: simulated providers keep per-task state
        if hasattr(provider, "begin_task"):
            provider.begin_task(task, prompt_name, trial)
        w = world.clone()
        ep = run_episode(provider, w, task, prompts[prompt_name])
        g = grade(task, ep, w)
        pin, pout = price_for(model, prices)
        cost = (ep["tokens_in"] * pin + ep["tokens_out"] * pout) / 1e6
        return {"model": model, "prompt": prompt_name, "task_id": task["id"], "category": task["category"],
                "trial": trial, **g, "cost_usd": round(cost, 6), **ep}

    rows: list[dict[str, Any]] = []
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, j) for j in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            rows.append(row)
            f.write(json.dumps(row) + "\n")
            if on_progress:
                on_progress(i, len(jobs))
    return rows
