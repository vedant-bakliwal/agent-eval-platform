import tempfile
import unittest
from pathlib import Path

from agenteval.env import ToolError
from agenteval.grading import grade
from agenteval.report import check_gates, compare, summarize, wilson
from agenteval.runner import run, run_episode
from agenteval.providers.base import ModelTurn, ToolCall
from agenteval.tasks import CATEGORIES, generate, load_suite

PROMPTS = {"baseline": "You are a support agent."}


class Suite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = Path(tempfile.mkdtemp())
        generate(cls.dir / "suite", per_category=8, seed=1)
        cls.world, cls.tasks = load_suite(cls.dir / "suite")

    def test_generates_every_category(self):
        self.assertEqual({t["category"] for t in self.tasks}, set(CATEGORIES))
        self.assertEqual(len(self.tasks), 80)

    def test_generation_is_deterministic(self):
        generate(self.dir / "again", per_category=8, seed=1)
        self.assertEqual((self.dir / "suite/tasks.jsonl").read_text(), (self.dir / "again/tasks.jsonl").read_text())

    def test_oracle_solves_every_task(self):
        rows = run((self.world, self.tasks), ["sim:oracle"], PROMPTS, 1, self.dir / "o.jsonl", workers=4)
        failed = [(r["task_id"], r["failure_all"]) for r in rows if not r["success"]]
        self.assertEqual(failed, [])

    def test_weaker_model_scores_lower_and_is_reproducible(self):
        a = summarize(run((self.world, self.tasks), ["sim:small-oss"], PROMPTS, 2, self.dir / "a.jsonl"))[0]
        b = summarize(run((self.world, self.tasks), ["sim:small-oss"], PROMPTS, 2, self.dir / "b.jsonl"))[0]
        self.assertLess(a["success_rate"], 0.95)
        self.assertEqual(a["success_rate"], b["success_rate"])
        self.assertLessEqual(a["pass_all_trials"], a["success_rate"])


class Grading(unittest.TestCase):
    def setUp(self):
        d = Path(tempfile.mkdtemp())
        generate(d, per_category=3, seed=2)
        self.world, self.tasks = load_suite(d)

    def task(self, category):
        return next(t for t in self.tasks if t["category"] == category)

    def test_saying_done_without_doing_it_fails(self):
        t = self.task("cancel")

        class Liar:
            name = "liar"
            def chat(self, system, messages, tools):
                return ModelTurn(text="Done, your order is cancelled!")

        w = self.world.clone()
        g = grade(t, run_episode(Liar(), w, t, ""), w)
        self.assertFalse(g["success"])
        self.assertEqual(g["failure"], "wrong_state")

    def test_forbidden_refund_is_caught_even_if_answer_is_good(self):
        t = self.task("out_of_policy")
        oid = t["reference"]["plan"][0]["args"]["summary"].split()[-1]

        class Refunder:
            name = "refunder"
            calls = 0
            def chat(self, system, messages, tools):
                self.calls += 1
                if self.calls == 1:
                    return ModelTurn("", [ToolCall("1", "issue_refund", {"order_id": oid, "amount": 1.0})])
                if self.calls == 2:
                    return ModelTurn("", [ToolCall("2", "escalate_to_human", {"summary": "x"})])
                return ModelTurn("Our team will follow up.")

        w = self.world.clone()
        g = grade(t, run_episode(Refunder(), w, t, ""), w)
        self.assertEqual(g["failure"], "forbidden_action")

    def test_hallucinated_tool_and_bad_args_lower_accuracy(self):
        t = self.task("order_status")
        oid = t["expect"]["required_tools"][0]["args"]["order_id"]
        script = [
            ModelTurn("", [ToolCall("1", "lookup_customer", {})]),
            ModelTurn("", [ToolCall("2", "get_shipping_status", {"order_id": "#123"})]),
            ModelTurn("", [ToolCall("3", "get_shipping_status", {"order_id": oid})]),
        ]

        class Sloppy:
            name = "sloppy"
            def chat(self, system, messages, tools):
                if script:
                    return script.pop(0)
                return ModelTurn("Tracking: " + messages[-1]["results"][0]["content"])

        w = self.world.clone()
        g = grade(t, run_episode(Sloppy(), w, t, ""), w)
        self.assertTrue(g["success"])
        self.assertEqual(g["hallucinated_tools"], 1)
        self.assertAlmostEqual(g["tool_call_accuracy"], 1 / 3, places=3)

    def test_flaky_tool_fails_once_then_works(self):
        t = self.task("flaky_tool_retry")
        w = self.world.clone()
        w.flaky = {"cancel_order": 1}
        oid = t["expect"]["required_tools"][0]["args"]["order_id"]
        with self.assertRaises(ToolError):
            w.call("cancel_order", {"order_id": oid})
        self.assertEqual(w.call("cancel_order", {"order_id": oid})["status"], "cancelled")

    def test_step_limit(self):
        t = self.task("cancel")

        class Looper:
            name = "looper"
            def chat(self, system, messages, tools):
                return ModelTurn("", [ToolCall("x", "search_catalog", {"query": "a"})])

        w = self.world.clone()
        ep = run_episode(Looper(), w, t, "")
        self.assertEqual(ep["status"], "step_limit")
        self.assertEqual(grade(t, ep, w)["failure"], "step_limit")


class Reporting(unittest.TestCase):
    def test_wilson_interval(self):
        lo, hi = wilson(90, 100)
        self.assertTrue(0.82 < lo < 0.84 and 0.94 < hi < 0.95)

    def test_compare_and_gates(self):
        rows = [
            {"model": "m", "prompt": "a", "task_id": "t1", "success": True},
            {"model": "m", "prompt": "b", "task_id": "t1", "success": False},
            {"model": "m", "prompt": "a", "task_id": "t2", "success": False},
            {"model": "m", "prompt": "b", "task_id": "t2", "success": True},
        ]
        c = compare(rows, "m / a", "m / b")
        self.assertEqual((c["regressions"], c["fixes"]), (["t1"], ["t2"]))
        s = [{"config": "x", "success_rate": 0.5, "cost_per_task": 0.02}]
        self.assertEqual(len(check_gates(s, {"success_rate": 0.8, "cost_per_task<=": 0.01})), 2)


if __name__ == "__main__":
    unittest.main()
