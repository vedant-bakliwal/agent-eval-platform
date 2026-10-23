"""Exercises the real-provider adapters against a local fake API (no network, no keys)."""
import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest import mock

from agenteval.providers import real
from agenteval.runner import run_episode
from agenteval.env import World

ORDER = {"ORD-1001": {"id": "ORD-1001", "status": "processing", "items": [], "total": 10.0,
                      "email": "a@example.com", "address": "x", "carrier": None, "tracking": None, "eta": None}}
TASK = {"id": "t", "category": "cancel", "user": "cancel ORD-1001", "setup": {},
        "expect": {"max_steps": 5}}


class Fake(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        Fake.seen.append((self.path, body))
        msgs = body["messages"]
        if self.path.endswith("/chat/completions"):
            done = any(m["role"] == "tool" for m in msgs)
            msg = ({"role": "assistant", "content": "Cancelled."} if done else
                   {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function",
                    "function": {"name": "cancel_order", "arguments": "{\"order_id\": \"ORD-1001\"}"}}]})
            out = {"choices": [{"message": msg}], "usage": {"prompt_tokens": 100, "completion_tokens": 10}}
        else:
            done = any(isinstance(m["content"], list) and m["content"][0].get("type") == "tool_result" for m in msgs)
            content = ([{"type": "text", "text": "Cancelled."}] if done else
                       [{"type": "tool_use", "id": "tu1", "name": "cancel_order", "input": {"order_id": "ORD-1001"}}])
            out = {"content": content, "usage": {"input_tokens": 120, "output_tokens": 12}}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.end_headers()
        self.wfile.write(data)


class Providers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = HTTPServer(("127.0.0.1", 0), Fake)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.url = f"http://127.0.0.1:{cls.srv.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def setUp(self):
        Fake.seen = []

    def episode(self, provider):
        w = World(books={}, orders=json.loads(json.dumps(ORDER)))
        ep = run_episode(provider, w, TASK, "sys")
        return ep, w

    def test_openai_compatible(self):
        with mock.patch.dict(os.environ, {"OLLAMA_BASE_URL": self.url + "/v1"}):
            p = real.make_provider("ollama:qwen3:8b")
        ep, w = self.episode(p)
        self.assertEqual(w.orders["ORD-1001"]["status"], "cancelled")
        self.assertEqual(ep["final"], "Cancelled.")
        self.assertEqual((ep["tokens_in"], ep["tokens_out"]), (200, 20))
        second = Fake.seen[1][1]["messages"]
        self.assertEqual(second[0]["role"], "system")
        self.assertEqual(second[-1], {"role": "tool", "tool_call_id": "c1", "content": json.dumps({"id": "ORD-1001", "status": "cancelled"})})
        self.assertEqual(Fake.seen[0][1]["model"], "qwen3:8b")

    def test_anthropic(self):
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test"}):
            p = real.make_provider("anthropic:claude-test")
        with mock.patch.object(real, "_post", side_effect=lambda url, body, headers, **k: real.__dict__["_orig"](self.url + "/v1/messages", body, headers)):
            ep, w = self.episode(p)
        self.assertEqual(w.orders["ORD-1001"]["status"], "cancelled")
        self.assertEqual((ep["tokens_in"], ep["tokens_out"]), (240, 24))
        last = Fake.seen[1][1]["messages"][-1]["content"][0]
        self.assertEqual((last["type"], last["tool_use_id"], last["is_error"]), ("tool_result", "tu1", False))

    def test_missing_key_is_a_clear_error(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(real.ProviderError):
                real.make_provider("anthropic:x")


real._orig = real._post

if __name__ == "__main__":
    unittest.main()
