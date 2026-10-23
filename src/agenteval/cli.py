"""agenteval: run agents against hundreds of tasks and see which model + prompt actually works."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import report
from .runner import run
from .tasks import generate, load_suite

ROOT = Path(__file__).resolve().parents[2]


def _prompts(names: str, prompt_dir: Path) -> dict[str, str]:
    out = {}
    for n in names.split(","):
        p = prompt_dir / f"{n.strip()}.txt"
        if not p.exists():
            sys.exit(f"prompt '{n}' not found in {prompt_dir}")
        out[n.strip()] = p.read_text()
    return out


def _prices(items: list[str]) -> dict[str, tuple[float, float]]:
    out = {}
    for it in items or []:
        model, _, vals = it.rpartition("=")
        a, b = vals.split(",")
        out[model] = (float(a), float(b))
    return out


def _gates(items: list[str]) -> dict[str, float]:
    out = {}
    for it in items or []:
        if "<=" in it:
            k, v = it.split("<=")
            out[k.strip() + "<="] = float(v)
        else:
            k, v = it.split(">=") if ">=" in it else it.split("=")
            out[k.strip()] = float(v)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="agenteval", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="build the task suite")
    g.add_argument("--out", default=str(ROOT / "suites/bookshop"))
    g.add_argument("--per-category", type=int, default=30)
    g.add_argument("--seed", type=int, default=7)

    r = sub.add_parser("run", help="run models × prompts × tasks × trials")
    r.add_argument("--suite", default=str(ROOT / "suites/bookshop"))
    r.add_argument("--models", required=True, help="comma list, e.g. sim:frontier,anthropic:claude-sonnet-4-5,ollama:qwen3:8b")
    r.add_argument("--prompts", default="baseline")
    r.add_argument("--prompt-dir", default=str(ROOT / "prompts"))
    r.add_argument("--trials", type=int, default=3)
    r.add_argument("--limit", type=int, help="only the first N tasks (for a cheap smoke run)")
    r.add_argument("--workers", type=int, default=8)
    r.add_argument("--price", action="append", help="model=in,out USD per 1M tokens")
    r.add_argument("--out", default="out")
    r.add_argument("--compare", action="append", help="'base config' vs 'candidate config', as BASE..CANDIDATE")
    r.add_argument("--gate", action="append", help="e.g. success_rate>=0.85, pass_all_trials>=0.7, cost_per_task<=0.01")

    rp = sub.add_parser("report", help="rebuild the dashboard from a results file")
    rp.add_argument("results")
    rp.add_argument("--out", default="out")
    rp.add_argument("--compare", action="append")
    rp.add_argument("--gate", action="append")

    a = ap.parse_args(argv)
    if a.cmd == "generate":
        n = generate(Path(a.out), a.per_category, a.seed)
        print(f"wrote {n} tasks to {a.out}")
        return 0

    out = Path(a.out)
    if a.cmd == "run":
        suite = load_suite(Path(a.suite))
        models = [m.strip() for m in a.models.split(",")]
        prompts = _prompts(a.prompts, Path(a.prompt_dir))
        total = len(models) * len(prompts) * len(suite[1][: a.limit] if a.limit else suite[1]) * a.trials

        def progress(i, n):
            if i % max(1, n // 20) == 0 or i == n:
                print(f"\r  {i}/{n} episodes", end="", file=sys.stderr, flush=True)

        print(f"running {total} episodes ({len(models)} models × {len(prompts)} prompts × {a.trials} trials)", file=sys.stderr)
        rows = run(suite, models, prompts, a.trials, out / "results.jsonl", a.workers, _prices(a.price), a.limit, progress)
        print(file=sys.stderr)
    else:
        rows = report.load_rows(Path(a.results))

    summary = report.summarize(rows)
    comps = []
    for c in a.compare or []:
        base, _, cand = c.partition("..")
        comps.append(report.compare(rows, base.strip(), cand.strip()))
    (out / "summary.json").write_text(json.dumps({"summary": summary, "comparisons": comps}, indent=2))
    report.write_html(rows, summary, out / "report.html", "Agent evaluation report",
                      comps, note=f"{len(rows):,} episodes · suite: bookshop support desk")

    w = max(len(s["config"]) for s in summary)
    print(f"\n{'config':<{w}}  success  pass^k  tool-acc  p95(s)  $/task")
    for s in summary:
        print(f"{s['config']:<{w}}  {s['success_rate']:7.1%}  {s['pass_all_trials']:6.1%}  {s['tool_call_accuracy']:8.1%}"
              f"  {s['latency_p95']:6.2f}  {s['cost_per_task']:.5f}")
    for c in comps:
        print(f"\n{c['candidate']} vs {c['base']}: {len(c['fixes'])} fixed, {len(c['regressions'])} regressed")
    print(f"\nreport: {out / 'report.html'}")

    problems = report.check_gates(summary, _gates(a.gate))
    if problems:
        print("\nGATE FAILED:\n  " + "\n  ".join(problems))
        return 1
    if a.gate:
        print("\ngate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
