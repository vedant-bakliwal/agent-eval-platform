"""Aggregates run results, compares configurations, enforces gates, and writes the HTML dashboard."""
from __future__ import annotations

import html
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .grading import FAILURE_ORDER


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def pct(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    i = (len(xs) - 1) * q
    lo, hi = math.floor(i), math.ceil(i)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(r["model"], r["prompt"])].append(r)
    out = []
    for (model, prompt), rs in groups.items():
        n = len(rs)
        k = sum(r["success"] for r in rs)
        by_task: dict[str, list[bool]] = defaultdict(list)
        for r in rs:
            by_task[r["task_id"]].append(r["success"])
        trials = max(len(v) for v in by_task.values())
        lat = [r["latency_s"] for r in rs]
        cost = sum(r["cost_usd"] for r in rs)
        calls = sum(r["tool_calls"] for r in rs)
        cats: dict[str, list[bool]] = defaultdict(list)
        for r in rs:
            cats[r["category"]].append(r["success"])
        out.append({
            "config": f"{model} / {prompt}", "model": model, "prompt": prompt,
            "episodes": n, "tasks": len(by_task), "trials": trials,
            "success_rate": k / n, "success_ci": wilson(k, n),
            # pass^k: the task succeeded on EVERY trial. This is the reliability number.
            "pass_all_trials": sum(all(v) for v in by_task.values()) / len(by_task),
            "pass_any_trial": sum(any(v) for v in by_task.values()) / len(by_task),
            "tool_call_accuracy": (sum(r["tool_call_accuracy"] * r["tool_calls"] for r in rs) / calls) if calls else 1.0,
            "required_recall": statistics.mean(r["required_recall"] for r in rs),
            "hallucinated_tools": sum(r["hallucinated_tools"] for r in rs),
            "avg_tool_calls": calls / n,
            "latency_p50": pct(lat, 0.5), "latency_p95": pct(lat, 0.95),
            "tokens_in_avg": statistics.mean(r["tokens_in"] for r in rs),
            "tokens_out_avg": statistics.mean(r["tokens_out"] for r in rs),
            "cost_total": cost, "cost_per_task": cost / n, "cost_per_success": (cost / k) if k else None,
            "failures": dict(Counter(r["failure"] for r in rs if r["failure"])),
            "by_category": {c: sum(v) / len(v) for c, v in sorted(cats.items())},
        })
    out.sort(key=lambda s: -s["success_rate"])
    return out


def compare(rows: list[dict[str, Any]], base: str, cand: str) -> dict[str, Any]:
    """Per-task diff between two configs ('model / prompt'). Majority vote across trials."""
    def votes(cfg):
        d: dict[str, list[bool]] = defaultdict(list)
        for r in rows:
            if f"{r['model']} / {r['prompt']}" == cfg:
                d[r["task_id"]].append(r["success"])
        return {t: sum(v) / len(v) >= 0.5 for t, v in d.items()}
    a, b = votes(base), votes(cand)
    shared = sorted(set(a) & set(b))
    return {"base": base, "candidate": cand, "tasks": len(shared),
            "regressions": [t for t in shared if a[t] and not b[t]],
            "fixes": [t for t in shared if not a[t] and b[t]]}


def check_gates(summary: list[dict[str, Any]], gates: dict[str, float]) -> list[str]:
    """gates like {"success_rate": 0.85, "pass_all_trials": 0.75, "cost_per_task<=": 0.01}."""
    problems = []
    for s in summary:
        for key, limit in gates.items():
            if key.endswith("<="):
                val = s[key[:-2]]
                if val is not None and val > limit:
                    problems.append(f"{s['config']}: {key[:-2]} {val:.4g} > {limit}")
            elif s[key] < limit:
                problems.append(f"{s['config']}: {key} {s[key]:.3f} < {limit}")
    return problems


# ---------------------------------------------------------------- HTML dashboard

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SERIES_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]


def _e(x: Any) -> str:
    return html.escape(str(x))


def _bar_rows(summary, key, fmt, ci_key=None, lo=0.0, hi=1.0) -> str:
    out = []
    for i, s in enumerate(summary):
        v = s[key]
        w = 0 if hi == lo else max(0.0, (v - lo) / (hi - lo)) * 100
        ci = ""
        tip = f"{s['config']}: {fmt(v)}"
        if ci_key:
            a, b = s[ci_key]
            ci = f"<span class='ci' style='left:{a*100:.2f}%;width:{(b-a)*100:.2f}%'></span>"
            tip += f" (95% CI {fmt(a)}–{fmt(b)})"
        out.append(
            f"<div class='brow' title='{_e(tip)}'><div class='blabel'>{_e(s['config'])}</div>"
            f"<div class='btrack'><span class='bar' style='width:{w:.2f}%;background:var(--s{i % 8})'></span>{ci}</div>"
            f"<div class='bval'>{fmt(v)}</div></div>")
    return "".join(out)


def _p(x: float) -> str:
    return f"{x*100:.1f}%"


def _usd(x: float | None) -> str:
    if x is None:
        return "–"
    return f"${x:.4f}" if x < 1 else f"${x:,.2f}"


def write_html(rows: list[dict[str, Any]], summary: list[dict[str, Any]], path: Path, title: str,
               comparisons: list[dict[str, Any]] | None = None, note: str = "") -> None:
    cats = sorted({c for s in summary for c in s["by_category"]})
    best = summary[0] if summary else None
    simulated = any(s["model"].startswith("sim:") for s in summary)

    tiles = ""
    if best:
        tiles = "".join(f"<div class='tile'><div class='tk'>{k}</div><div class='tv'>{v}</div><div class='ts'>{sub}</div></div>"
                        for k, v, sub in [
                            ("Episodes run", f"{len(rows):,}", f"{best['tasks']} tasks × {len(summary)} configs × {best['trials']} trials"),
                            ("Best success rate", _p(best["success_rate"]), _e(best["config"])),
                            ("Best pass^k (every trial)", _p(max(s["pass_all_trials"] for s in summary)), "task solved on all trials"),
                            ("Cheapest per success", _usd(min((s["cost_per_success"] for s in summary if s["cost_per_success"]), default=None)), "USD per solved task"),
                        ])

    max_cost = max((s["cost_per_task"] for s in summary), default=0) or 1
    max_lat = max((s["latency_p95"] for s in summary), default=0) or 1

    heat_head = "".join(f"<th>{_e(c.replace('_', ' '))}</th>" for c in cats)
    heat_rows = ""
    for s in summary:
        cells = ""
        for c in cats:
            v = s["by_category"].get(c)
            if v is None:
                cells += "<td>–</td>"
                continue
            step = sum(v >= t for t in (0.5, 0.7, 0.85, 0.95))
            cells += f"<td class='h{step}' title='{_e(s['config'])} · {_e(c)}: {_p(v)}'>{v*100:.0f}</td>"
        heat_rows += f"<tr><th class='rl'>{_e(s['config'])}</th>{cells}</tr>"

    fail_keys = [f for f in FAILURE_ORDER if any(f in s["failures"] for s in summary)]
    fail_head = "".join(f"<th>{_e(f.replace('_', ' '))}</th>" for f in fail_keys)
    fail_rows = "".join(
        f"<tr><th class='rl'>{_e(s['config'])}</th>" +
        "".join(f"<td>{s['failures'].get(f, 0)}</td>" for f in fail_keys) +
        f"<td><b>{sum(s['failures'].values())}</b></td></tr>" for s in summary)

    metric_rows = "".join(
        f"<tr><th class='rl'>{_e(s['config'])}</th><td>{_p(s['success_rate'])}</td><td>{_p(s['pass_all_trials'])}</td>"
        f"<td>{_p(s['tool_call_accuracy'])}</td><td>{_p(s['required_recall'])}</td><td>{s['hallucinated_tools']}</td>"
        f"<td>{s['avg_tool_calls']:.1f}</td><td>{s['latency_p50']:.2f}s</td><td>{s['latency_p95']:.2f}s</td>"
        f"<td>{s['tokens_in_avg']:,.0f}</td><td>{s['tokens_out_avg']:,.0f}</td><td>{_usd(s['cost_per_task'])}</td>"
        f"<td>{_usd(s['cost_per_success'])}</td></tr>" for s in summary)

    failed = [r for r in rows if not r["success"]]
    failed.sort(key=lambda r: (r["model"], r["prompt"], r["task_id"], r["trial"]))
    samples = ""
    for r in failed[:60]:
        steps = " → ".join(f"{c['name']}({', '.join(f'{k}={v}' for k, v in c['args'].items())}){'' if c['ok'] else ' ✗'}"
                           for c in r["trace"]) or "(no tool calls)"
        samples += (f"<details><summary><code>{_e(r['task_id'])}</code> · {_e(r['model'])} / {_e(r['prompt'])} · "
                    f"trial {r['trial']} · <b>{_e(r['failure'])}</b></summary>"
                    f"<div class='tr'><div><b>Tools:</b> {_e(steps)}</div><div><b>Reply:</b> {_e(r['final'] or '(none)')}</div>"
                    f"{'<div><b>Missing facts:</b> ' + _e(', '.join(r['missing_facts'])) + '</div>' if r['missing_facts'] else ''}</div></details>")

    comp_html = ""
    for c in comparisons or []:
        comp_html += (f"<div class='card'><h3>{_e(c['candidate'])} vs {_e(c['base'])}</h3>"
                      f"<p>{len(c['fixes'])} tasks fixed, <b>{len(c['regressions'])} regressed</b> (of {c['tasks']}, majority vote across trials).</p>"
                      f"{'<p class=muted>Regressed: ' + _e(', '.join(c['regressions'][:40])) + '</p>' if c['regressions'] else ''}</div>")

    legend = "".join(f"<span class='lg'><span class='sw' style='background:var(--s{i % 8})'></span>{_e(s['config'])}</span>"
                     for i, s in enumerate(summary))
    sim_banner = ("<div class='banner'>Simulated models (sim:*) are offline stand-ins used to test the platform. "
                  "Their numbers describe the simulator, not real models. Run with anthropic:, openai:, ollama:, vllm: "
                  "or bedrock: models for real results.</div>") if simulated else ""

    css_vars_l = ";".join(f"--s{i}:{c}" for i, c in enumerate(SERIES))
    css_vars_d = ";".join(f"--s{i}:{c}" for i, c in enumerate(SERIES_DARK))
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_e(title)}</title><style>
:root{{color-scheme:light;--bg:#fcfcfb;--card:#ffffff;--ink:#0b0b0b;--ink2:#52514e;--line:#e4e2dc;--track:#efeee9;
--h0:#f3f6fb;--h1:#cfe0f6;--h2:#94bdee;--h3:#4f8fdc;--h4:#1d5fb3;--h3i:#fff;{css_vars_l}}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{color-scheme:dark;--bg:#151514;--card:#1d1d1c;--ink:#f0efec;--ink2:#c3c2b7;--line:#33322f;--track:#2a2a28;
--h0:#1f2530;--h1:#21395c;--h2:#2a5591;--h3:#3a76c4;--h4:#6aa5ec;{css_vars_d}}}}}
:root[data-theme="dark"]{{color-scheme:dark;--bg:#151514;--card:#1d1d1c;--ink:#f0efec;--ink2:#c3c2b7;--line:#33322f;--track:#2a2a28;
--h0:#1f2530;--h1:#21395c;--h2:#2a5591;--h3:#3a76c4;--h4:#6aa5ec;{css_vars_d}}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}}
main{{max-width:1180px;margin:0 auto;padding:24px 16px 64px}} h1{{font-size:22px;margin:0 0 4px}} h2{{font-size:16px;margin:28px 0 10px}} h3{{font-size:14px;margin:0 0 6px}}
.muted{{color:var(--ink2)}} .banner{{border:1px solid var(--line);border-left:4px solid var(--s3);background:var(--card);padding:10px 12px;border-radius:8px;margin:14px 0}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin-top:16px}}
.tile,.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}} .tk{{color:var(--ink2);font-size:12px}} .tv{{font-size:26px;font-weight:650;margin:2px 0}} .ts{{color:var(--ink2);font-size:12px;overflow-wrap:anywhere}}
.grid2{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,460px),1fr));gap:12px}}
.brow{{display:grid;grid-template-columns:minmax(120px,38%) 1fr 64px;gap:8px;align-items:center;margin:6px 0}} .blabel{{font-size:12px;color:var(--ink2);overflow-wrap:anywhere}}
.btrack{{position:relative;height:14px;background:var(--track);border-radius:4px}} .bar{{position:absolute;left:0;top:0;bottom:0;border-radius:0 4px 4px 0}}
.ci{{position:absolute;top:6px;height:2px;background:var(--ink)}} .bval{{font-variant-numeric:tabular-nums;text-align:right;font-size:12px}}
.lgs{{display:flex;flex-wrap:wrap;gap:12px;margin:4px 0 8px}} .lg{{font-size:12px;color:var(--ink2);display:flex;gap:6px;align-items:center}} .sw{{width:10px;height:10px;border-radius:2px}}
.scroll{{overflow-x:auto}} table{{border-collapse:collapse;width:100%;font-size:12px;font-variant-numeric:tabular-nums}} th,td{{padding:6px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}
th{{color:var(--ink2);font-weight:600}} th.rl{{text-align:left;color:var(--ink);font-weight:500}} thead th{{text-align:right}} thead th:first-child{{text-align:left}}
.h0{{background:var(--h0)}} .h1{{background:var(--h1)}} .h2{{background:var(--h2)}} .h3{{background:var(--h3);color:#fff}} .h4{{background:var(--h4);color:#fff}}
details{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px 10px;margin:6px 0}} summary{{cursor:pointer;font-size:13px}} .tr{{margin-top:6px;font-size:12px;display:grid;gap:4px;overflow-wrap:anywhere}}
code{{font-size:12px}}
</style></head><body><main>
<h1>{_e(title)}</h1><div class="muted">{_e(note)}</div>{sim_banner}
<div class="tiles">{tiles}</div>
<h2>Success rate</h2><div class="card"><div class="muted" style="font-size:12px">Share of episodes fully solved: right world state, right facts in the reply, no forbidden action. Black line = 95% confidence interval.</div>{_bar_rows(summary, 'success_rate', _p, 'success_ci')}</div>
<div class="grid2" style="margin-top:12px">
<div class="card"><h3>Reliability: pass^k</h3><div class="muted" style="font-size:12px">Tasks solved on every one of {best['trials'] if best else 0} trials. The gap from success rate is flakiness.</div>{_bar_rows(summary, 'pass_all_trials', _p)}</div>
<div class="card"><h3>Tool-call accuracy</h3><div class="muted" style="font-size:12px">Calls naming a real tool with arguments the tool accepted.</div>{_bar_rows(summary, 'tool_call_accuracy', _p)}</div>
<div class="card"><h3>Cost per task (USD)</h3>{_bar_rows(summary, 'cost_per_task', _usd, None, 0, max_cost)}</div>
<div class="card"><h3>Latency p95 per task (s)</h3>{_bar_rows(summary, 'latency_p95', lambda x: f'{x:.2f}s', None, 0, max_lat)}</div>
</div>
<div class="lgs">{legend}</div>
<h2>Success by task category (%)</h2><div class="card scroll"><table><thead><tr><th>config</th>{heat_head}</tr></thead><tbody>{heat_rows}</tbody></table></div>
<h2>Why episodes failed</h2><div class="card scroll"><div class="muted" style="font-size:12px;margin-bottom:6px">Primary failure reason per failed episode.</div><table><thead><tr><th>config</th>{fail_head}<th>total</th></tr></thead><tbody>{fail_rows}</tbody></table></div>
{('<h2>Comparisons</h2>' + comp_html) if comp_html else ''}
<h2>All metrics</h2><div class="card scroll"><table><thead><tr><th>config</th><th>success</th><th>pass^k</th><th>tool acc.</th><th>req. recall</th><th>halluc. tools</th><th>calls/task</th><th>p50</th><th>p95</th><th>tokens in</th><th>tokens out</th><th>$/task</th><th>$/success</th></tr></thead><tbody>{metric_rows}</tbody></table></div>
<h2>Failed episodes (first 60)</h2>{samples or '<p class=muted>None.</p>'}
</main></body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc)
