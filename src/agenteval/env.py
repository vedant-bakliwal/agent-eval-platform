"""A small, deterministic world the agent works in: an online bookshop's support desk.

Every task gets a fresh copy of the world, so runs never leak state into each other.
Tools mutate the world; the grader checks the world afterwards (did the order really
get cancelled?) instead of trusting what the agent says it did.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable

TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "search_catalog",
        "description": "Search books by title or author. Returns up to 5 matches with id, title, author, price and stock.",
        "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    },
    {
        "name": "get_order",
        "description": "Look up an order by id (format ORD-1234). Returns status, items, total and customer email.",
        "input_schema": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]},
    },
    {
        "name": "get_shipping_status",
        "description": "Get carrier, tracking number and estimated delivery date for a shipped order.",
        "input_schema": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]},
    },
    {
        "name": "cancel_order",
        "description": "Cancel an order. Only orders with status 'processing' can be cancelled.",
        "input_schema": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}, "reason": {"type": "string"}},
            "required": ["order_id"],
        },
    },
    {
        "name": "issue_refund",
        "description": "Refund a delivered order, fully or partly. amount must not exceed the order total.",
        "input_schema": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}, "amount": {"type": "number"}},
            "required": ["order_id", "amount"],
        },
    },
    {
        "name": "update_address",
        "description": "Change the shipping address of an order that has not shipped yet.",
        "input_schema": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}, "address": {"type": "string"}},
            "required": ["order_id", "address"],
        },
    },
    {
        "name": "escalate_to_human",
        "description": "Hand the conversation to a human agent with a short summary. Use for requests outside policy.",
        "input_schema": {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]},
    },
]

TOOL_NAMES = {t["name"] for t in TOOL_SPECS}


class ToolError(Exception):
    """A tool refused the call. The message goes back to the agent, like a real API error."""


@dataclass
class World:
    books: dict[str, dict[str, Any]]
    orders: dict[str, dict[str, Any]]
    refunds: list[dict[str, Any]] = field(default_factory=list)
    escalations: list[str] = field(default_factory=list)
    # Fault injection: tool name -> number of times it fails before it works.
    flaky: dict[str, int] = field(default_factory=dict)

    def clone(self) -> "World":
        return copy.deepcopy(self)

    # --- tools -----------------------------------------------------------
    def call(self, name: str, args: dict[str, Any]) -> Any:
        if name not in TOOL_NAMES:
            raise ToolError(f"unknown tool '{name}'")
        if self.flaky.get(name, 0) > 0:
            self.flaky[name] -= 1
            raise ToolError("503 Service Unavailable: please retry")
        handler: Callable[..., Any] = getattr(self, f"_t_{name}")
        try:
            return handler(**args)
        except TypeError as exc:  # wrong or missing arguments
            raise ToolError(f"bad arguments for {name}: {exc}") from None

    def _order(self, order_id: str) -> dict[str, Any]:
        order = self.orders.get(str(order_id).strip().upper())
        if order is None:
            raise ToolError(f"order {order_id} not found")
        return order

    def _t_search_catalog(self, query: str) -> list[dict[str, Any]]:
        q = query.lower()
        hits = [dict(id=i, **b) for i, b in self.books.items() if q in b["title"].lower() or q in b["author"].lower()]
        return hits[:5]

    def _t_get_order(self, order_id: str) -> dict[str, Any]:
        o = self._order(order_id)
        return {k: o[k] for k in ("id", "status", "items", "total", "email")}

    def _t_get_shipping_status(self, order_id: str) -> dict[str, Any]:
        o = self._order(order_id)
        if o["status"] not in ("shipped", "delivered"):
            raise ToolError(f"order {o['id']} has not shipped (status: {o['status']})")
        return {"carrier": o["carrier"], "tracking": o["tracking"], "eta": o["eta"]}

    def _t_cancel_order(self, order_id: str, reason: str = "") -> dict[str, Any]:
        o = self._order(order_id)
        if o["status"] != "processing":
            raise ToolError(f"cannot cancel order {o['id']}: status is {o['status']}")
        o["status"] = "cancelled"
        return {"id": o["id"], "status": "cancelled"}

    def _t_issue_refund(self, order_id: str, amount: float) -> dict[str, Any]:
        o = self._order(order_id)
        if o["status"] != "delivered":
            raise ToolError(f"only delivered orders can be refunded (status: {o['status']})")
        amount = round(float(amount), 2)
        already = sum(r["amount"] for r in self.refunds if r["order_id"] == o["id"])
        if amount <= 0 or amount + already > o["total"] + 1e-9:
            raise ToolError(f"refund {amount} exceeds refundable balance {round(o['total'] - already, 2)}")
        self.refunds.append({"order_id": o["id"], "amount": amount})
        return {"order_id": o["id"], "refunded": amount}

    def _t_update_address(self, order_id: str, address: str) -> dict[str, Any]:
        o = self._order(order_id)
        if o["status"] != "processing":
            raise ToolError(f"address can only change before shipping (status: {o['status']})")
        o["address"] = address
        return {"id": o["id"], "address": address}

    def _t_escalate_to_human(self, summary: str) -> dict[str, Any]:
        self.escalations.append(summary)
        return {"ticket": f"HUM-{len(self.escalations):03d}"}
