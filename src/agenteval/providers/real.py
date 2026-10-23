"""Real model backends. Standard library only (urllib), except Bedrock, which needs boto3.

  anthropic:<model>   Anthropic Messages API            ANTHROPIC_API_KEY
  openai:<model>      OpenAI Chat Completions           OPENAI_API_KEY (OPENAI_BASE_URL optional)
  ollama:<model>      Ollama's OpenAI-compatible API    OLLAMA_BASE_URL (default http://localhost:11434/v1)
  vllm:<model>        vLLM's OpenAI-compatible server   VLLM_BASE_URL (default http://localhost:8000/v1)
  bedrock:<model-id>  Amazon Bedrock Converse API       standard AWS credentials + AWS_REGION
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from .base import ModelTurn, ProviderError, ToolCall


def _post(url: str, body: dict[str, Any], headers: dict[str, str], timeout: float = 120, retries: int = 3) -> dict[str, Any]:
    data = json.dumps(body).encode()
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=data, headers={"content-type": "application/json", **headers})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:300]
            if exc.code in (429, 500, 502, 503, 529) and attempt < retries:
                time.sleep(2 ** attempt)
                continue
            raise ProviderError(f"HTTP {exc.code} from {url}: {detail}") from None
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt < retries:
                time.sleep(2 ** attempt)
                continue
            raise ProviderError(f"cannot reach {url}: {exc}") from None
    raise ProviderError("unreachable")


class AnthropicProvider:
    def __init__(self, model: str, max_tokens: int = 1024):
        self.name, self.model, self.max_tokens = f"anthropic:{model}", model, max_tokens
        self.key = os.environ.get("ANTHROPIC_API_KEY") or ""
        if not self.key:
            raise ProviderError("ANTHROPIC_API_KEY is not set")

    def chat(self, system, messages, tools) -> ModelTurn:
        msgs: list[dict[str, Any]] = []
        for m in messages:
            if m["role"] == "user":
                msgs.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                content: list[dict[str, Any]] = [{"type": "text", "text": m["text"]}] if m["text"] else []
                content += [{"type": "tool_use", "id": c["id"], "name": c["name"], "input": c["args"]} for c in m["tool_calls"]]
                msgs.append({"role": "assistant", "content": content})
            else:
                msgs.append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": r["id"], "content": r["content"], "is_error": r["is_error"]}
                    for r in m["results"]]})
        body = {"model": self.model, "max_tokens": self.max_tokens, "system": system, "messages": msgs, "tools": tools}
        out = _post("https://api.anthropic.com/v1/messages", body,
                    {"x-api-key": self.key, "anthropic-version": "2023-06-01"})
        text = "".join(b.get("text", "") for b in out["content"] if b["type"] == "text")
        calls = [ToolCall(b["id"], b["name"], b.get("input") or {}) for b in out["content"] if b["type"] == "tool_use"]
        u = out.get("usage", {})
        return ModelTurn(text, calls, u.get("input_tokens", 0), u.get("output_tokens", 0))


class OpenAICompatProvider:
    def __init__(self, prefix: str, model: str, base_url: str, key: str):
        self.name, self.model, self.base_url, self.key = f"{prefix}:{model}", model, base_url.rstrip("/"), key

    def chat(self, system, messages, tools) -> ModelTurn:
        msgs: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for m in messages:
            if m["role"] == "user":
                msgs.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                entry: dict[str, Any] = {"role": "assistant", "content": m["text"] or None}
                if m["tool_calls"]:
                    entry["tool_calls"] = [{"id": c["id"], "type": "function",
                                            "function": {"name": c["name"], "arguments": json.dumps(c["args"])}}
                                           for c in m["tool_calls"]]
                msgs.append(entry)
            else:
                msgs += [{"role": "tool", "tool_call_id": r["id"], "content": r["content"]} for r in m["results"]]
        oa_tools = [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                                       "parameters": t["input_schema"]}} for t in tools]
        headers = {"authorization": f"Bearer {self.key}"} if self.key else {}
        out = _post(f"{self.base_url}/chat/completions", {"model": self.model, "messages": msgs, "tools": oa_tools}, headers)
        msg = out["choices"][0]["message"]
        calls = []
        for c in msg.get("tool_calls") or []:
            raw = c["function"].get("arguments") or "{}"
            try:
                args = json.loads(raw)
            except json.JSONDecodeError:
                args = {"__unparseable__": raw}  # graded as a bad tool call, not a crash
            calls.append(ToolCall(c["id"], c["function"]["name"], args))
        u = out.get("usage") or {}
        return ModelTurn(msg.get("content") or "", calls, u.get("prompt_tokens", 0), u.get("completion_tokens", 0))


class BedrockProvider:
    def __init__(self, model_id: str):
        try:
            import boto3  # type: ignore
        except ImportError:
            raise ProviderError("bedrock: needs boto3 (pip install 'agenteval[bedrock]')") from None
        self.name, self.model = f"bedrock:{model_id}", model_id
        self.client = boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_REGION", "us-east-1"))

    def chat(self, system, messages, tools) -> ModelTurn:
        msgs: list[dict[str, Any]] = []
        for m in messages:
            if m["role"] == "user":
                msgs.append({"role": "user", "content": [{"text": m["content"]}]})
            elif m["role"] == "assistant":
                content = [{"text": m["text"]}] if m["text"] else []
                content += [{"toolUse": {"toolUseId": c["id"], "name": c["name"], "input": c["args"]}} for c in m["tool_calls"]]
                msgs.append({"role": "assistant", "content": content})
            else:
                msgs.append({"role": "user", "content": [
                    {"toolResult": {"toolUseId": r["id"], "content": [{"text": r["content"]}],
                                    "status": "error" if r["is_error"] else "success"}} for r in m["results"]]})
        tool_cfg = {"tools": [{"toolSpec": {"name": t["name"], "description": t["description"],
                                            "inputSchema": {"json": t["input_schema"]}}} for t in tools]}
        try:
            out = self.client.converse(modelId=self.model, system=[{"text": system}], messages=msgs, toolConfig=tool_cfg)
        except Exception as exc:  # botocore errors
            raise ProviderError(f"bedrock: {exc}") from None
        parts = out["output"]["message"]["content"]
        text = "".join(p.get("text", "") for p in parts)
        calls = [ToolCall(p["toolUse"]["toolUseId"], p["toolUse"]["name"], p["toolUse"].get("input") or {})
                 for p in parts if "toolUse" in p]
        u = out.get("usage", {})
        return ModelTurn(text, calls, u.get("inputTokens", 0), u.get("outputTokens", 0))


def make_provider(spec: str):
    from .simulated import SimulatedProvider

    prefix, _, model = spec.partition(":")
    if not model:
        raise ValueError(f"model spec must look like provider:model, got '{spec}'")
    if prefix == "sim":
        return SimulatedProvider(model)
    if prefix == "anthropic":
        return AnthropicProvider(model)
    if prefix == "openai":
        return OpenAICompatProvider("openai", model, os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                                    os.environ.get("OPENAI_API_KEY", ""))
    if prefix == "ollama":
        return OpenAICompatProvider("ollama", model, os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"), "")
    if prefix == "vllm":
        return OpenAICompatProvider("vllm", model, os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
                                    os.environ.get("VLLM_API_KEY", ""))
    if prefix == "bedrock":
        return BedrockProvider(model)
    raise ValueError(f"unknown provider '{prefix}'")
