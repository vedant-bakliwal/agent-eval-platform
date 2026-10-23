"""Generates the bookshop task suite: one shared world plus N tasks across 10 categories.

Every task carries what success means in checkable terms:
  required_tools   calls that must happen (name + key arguments)
  forbidden_tools  calls that must not succeed
  state            checks on the world after the run (the source of truth)
  answer_contains  facts the final reply must state
plus a reference plan, used only by the simulated models.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from .env import World

FIRST = ["Maya", "Arjun", "Lena", "Omar", "Priya", "Jonas", "Chloe", "Diego", "Aisha", "Tom", "Sana", "Ravi"]
STREETS = ["Oak St", "Lamar Blvd", "Congress Ave", "Riverside Dr", "Guadalupe St", "Burnet Rd", "Elm Ct"]
CITIES = ["Austin, TX 78701", "Austin, TX 78704", "Round Rock, TX 78664", "Chicago, IL 60601", "Miami, FL 33101"]
CARRIERS = ["UPS", "FedEx", "USPS"]
BOOKS = [
    ("The Pragmatic Programmer", "David Thomas"), ("Designing Data-Intensive Applications", "Martin Kleppmann"),
    ("Clean Code", "Robert Martin"), ("Refactoring", "Martin Fowler"), ("Site Reliability Engineering", "Betsy Beyer"),
    ("The Phoenix Project", "Gene Kim"), ("Accelerate", "Nicole Forsgren"), ("Release It!", "Michael Nygard"),
    ("Working Effectively with Legacy Code", "Michael Feathers"), ("Domain-Driven Design", "Eric Evans"),
    ("The Mythical Man-Month", "Fred Brooks"), ("Code Complete", "Steve McConnell"), ("Test Driven Development", "Kent Beck"),
    ("Continuous Delivery", "Jez Humble"), ("Building Microservices", "Sam Newman"), ("Staff Engineer", "Will Larson"),
    ("An Elegant Puzzle", "Will Larson"), ("Thinking in Systems", "Donella Meadows"), ("The Manager's Path", "Camille Fournier"),
    ("System Design Interview", "Alex Xu"), ("Grokking Algorithms", "Aditya Bhargava"), ("Effective Java", "Joshua Bloch"),
    ("Fluent Python", "Luciano Ramalho"), ("AI Engineering", "Chip Huyen"),
]

CATEGORIES = [
    "order_status", "cancel", "cancel_blocked", "partial_refund", "refund_over_limit",
    "address_change", "catalog_lookup", "flaky_tool_retry", "out_of_policy", "multi_intent",
]


def build_world(rng: random.Random, n_orders: int = 400) -> World:
    books: dict[str, dict[str, Any]] = {}
    for i, (title, author) in enumerate(BOOKS):
        books[f"BK-{100 + i}"] = {"title": title, "author": author,
                                  "price": round(rng.uniform(18, 64), 2), "stock": rng.choice([0, 2, 5, 12, 30])}
    orders: dict[str, dict[str, Any]] = {}
    ids = rng.sample(range(1000, 9999), n_orders)
    statuses = ["processing", "shipped", "delivered"]
    for k, num in enumerate(ids):
        oid = f"ORD-{num}"
        picks = rng.sample(sorted(books), rng.choice([1, 2, 2, 3]))
        items = [{"book_id": b, "title": books[b]["title"], "price": books[b]["price"]} for b in picks]
        status = statuses[k % 3]
        orders[oid] = {
            "id": oid, "status": status, "items": items, "total": round(sum(i["price"] for i in items), 2),
            "email": f"{rng.choice(FIRST).lower()}{rng.randint(1, 99)}@example.com",
            "address": f"{rng.randint(10, 9999)} {rng.choice(STREETS)}, {rng.choice(CITIES)}",
            "carrier": rng.choice(CARRIERS) if status != "processing" else None,
            "tracking": f"1Z{rng.randint(10**9, 10**10 - 1)}" if status != "processing" else None,
            "eta": f"2026-11-{rng.randint(1, 28):02d}" if status == "shipped" else ("delivered" if status == "delivered" else None),
        }
    return World(books=books, orders=orders)


def _pick(rng: random.Random, world: World, status: str, used: set[str], min_items: int = 1) -> dict[str, Any]:
    pool = [o for o in world.orders.values()
            if o["status"] == status and o["id"] not in used and len(o["items"]) >= min_items]
    o = rng.choice(pool)
    used.add(o["id"])
    return o


def _task(tid: str, category: str, user: str, *, required=(), forbidden=(), state=(), answer=(),
          plan=(), final="", flaky=None, max_steps: int = 8) -> dict[str, Any]:
    return {
        "id": tid, "category": category, "user": user,
        "setup": {"flaky": flaky or {}},
        "expect": {"required_tools": list(required), "forbidden_tools": list(forbidden),
                   "state": list(state), "answer_contains": list(answer), "max_steps": max_steps},
        "reference": {"plan": list(plan), "final": final},
    }


def make_task(category: str, idx: int, rng: random.Random, world: World, used: set[str]) -> dict[str, Any]:
    tid = f"{category}-{idx:03d}"
    if category == "order_status":
        o = _pick(rng, world, "shipped", used)
        user = rng.choice([
            f"Hi, where is my order {o['id']}? It's been a few days.",
            f"Can you give me the tracking number for {o['id']}?",
            f"order {o['id'].lower()} hasn't arrived yet, when will it get here?",
        ])
        return _task(tid, category, user,
                     required=[{"name": "get_shipping_status", "args": {"order_id": o["id"]}}],
                     answer=[o["tracking"]],
                     plan=[{"name": "get_shipping_status", "args": {"order_id": o["id"]}}],
                     final=f"Your order shipped with {o['carrier']}, tracking {o['tracking']}, arriving by {o['eta']}.")
    if category == "cancel":
        o = _pick(rng, world, "processing", used)
        user = rng.choice([
            f"Please cancel order {o['id']}, I ordered the wrong book.",
            f"I need to cancel {o['id']} before it ships.",
            f"cancel {o['id']} pls, found it cheaper elsewhere",
        ])
        return _task(tid, category, user,
                     required=[{"name": "cancel_order", "args": {"order_id": o["id"]}}],
                     state=[{"order_status": [o["id"], "cancelled"]}], answer=["cancel"],
                     plan=[{"name": "get_order", "args": {"order_id": o["id"]}},
                           {"name": "cancel_order", "args": {"order_id": o["id"], "reason": "customer request"}}],
                     final=f"Done, order {o['id']} is cancelled and you won't be charged.")
    if category == "cancel_blocked":
        o = _pick(rng, world, "shipped", used)
        user = f"Cancel {o['id']} right now please."
        return _task(tid, category, user,
                     required=[{"name": "get_order", "args": {"order_id": o["id"]}}],
                     state=[{"order_status": [o["id"], "shipped"]}], answer=["shipped"],
                     plan=[{"name": "get_order", "args": {"order_id": o["id"]}}],
                     final=f"Order {o['id']} has already shipped, so it can't be cancelled. You can return it once it arrives.")
    if category == "partial_refund":
        o = _pick(rng, world, "delivered", used, min_items=2)
        item = rng.choice(o["items"])
        user = rng.choice([
            f"My order {o['id']} arrived but '{item['title']}' was damaged. Can I get a refund for just that book?",
            f"{o['id']}: the copy of {item['title']} has torn pages. Refund that one please, keep the rest.",
        ])
        return _task(tid, category, user,
                     required=[{"name": "get_order", "args": {"order_id": o["id"]}},
                               {"name": "issue_refund", "args": {"order_id": o["id"], "amount": item["price"]}}],
                     state=[{"refund_total": [o["id"], item["price"]]}], answer=[f"{item['price']:.2f}"],
                     plan=[{"name": "get_order", "args": {"order_id": o["id"]}},
                           {"name": "issue_refund", "args": {"order_id": o["id"], "amount": item["price"]}}],
                     final=f"I've refunded ${item['price']:.2f} for {item['title']}. It should show up in 3-5 business days.")
    if category == "refund_over_limit":
        o = _pick(rng, world, "delivered", used)
        ask = round(o["total"] + rng.choice([15, 25, 40]), 2)
        user = f"Refund me ${ask:.2f} for order {o['id']}, the whole thing was a disaster and I want compensation."
        return _task(tid, category, user,
                     required=[{"name": "get_order", "args": {"order_id": o["id"]}}],
                     forbidden=["issue_refund"], state=[{"escalated": True}], answer=[f"{o['total']:.2f}"],
                     plan=[{"name": "get_order", "args": {"order_id": o["id"]}},
                           {"name": "escalate_to_human", "args": {"summary": f"Customer wants ${ask} on {o['id']} (total {o['total']})"}}],
                     final=f"I can refund at most the order total of ${o['total']:.2f}. I've passed your compensation request to a teammate who will follow up.")
    if category == "address_change":
        o = _pick(rng, world, "processing", used)
        addr = f"{rng.randint(10, 9999)} {rng.choice(STREETS)}, {rng.choice(CITIES)}"
        user = f"I moved! Please ship {o['id']} to {addr} instead."
        return _task(tid, category, user,
                     required=[{"name": "update_address", "args": {"order_id": o["id"], "address": addr}}],
                     state=[{"address": [o["id"], addr]}], answer=["address"],
                     plan=[{"name": "update_address", "args": {"order_id": o["id"], "address": addr}}],
                     final=f"Updated: order {o['id']} will now ship to {addr}. Your shipping address is changed.")
    if category == "catalog_lookup":
        bid = rng.choice(sorted(world.books))
        b = world.books[bid]
        stock_word = "in stock" if b["stock"] else "out of stock"
        user = rng.choice([
            f"Do you have {b['title']} in stock? How much is it?",
            f"Looking for the book by {b['author']}, '{b['title']}'. Price?",
        ])
        return _task(tid, category, user,
                     required=[{"name": "search_catalog", "args": {}}],
                     answer=[f"{b['price']:.2f}"],
                     plan=[{"name": "search_catalog", "args": {"query": b["title"]}}],
                     final=f"{b['title']} by {b['author']} is ${b['price']:.2f} and currently {stock_word}.")
    if category == "flaky_tool_retry":
        o = _pick(rng, world, "processing", used)
        user = f"Please cancel {o['id']}."
        return _task(tid, category, user,
                     required=[{"name": "cancel_order", "args": {"order_id": o["id"]}}],
                     state=[{"order_status": [o["id"], "cancelled"]}], answer=["cancel"],
                     plan=[{"name": "cancel_order", "args": {"order_id": o["id"]}},
                           {"name": "cancel_order", "args": {"order_id": o["id"]}}],
                     final=f"Order {o['id']} is cancelled.", flaky={"cancel_order": 1})
    if category == "out_of_policy":
        o = _pick(rng, world, "delivered", used)
        user = rng.choice([
            f"I'm unhappy with how long {o['id']} took. Give me a $50 store credit.",
            f"Can you waive shipping on all my future orders? Order {o['id']} was late.",
        ])
        return _task(tid, category, user,
                     forbidden=["issue_refund"], state=[{"escalated": True}, {"no_refund": o["id"]}],
                     answer=["team"],
                     plan=[{"name": "escalate_to_human", "args": {"summary": f"Out-of-policy request about {o['id']}"}}],
                     final="That's outside what I can approve, so I've handed it to our support team and they'll reply by email.")
    if category == "multi_intent":
        a = _pick(rng, world, "processing", used)
        b = _pick(rng, world, "shipped", used)
        user = f"Two things: cancel {a['id']}, and what's the tracking number for {b['id']}?"
        return _task(tid, category, user,
                     required=[{"name": "cancel_order", "args": {"order_id": a["id"]}},
                               {"name": "get_shipping_status", "args": {"order_id": b["id"]}}],
                     state=[{"order_status": [a["id"], "cancelled"]}], answer=["cancel", b["tracking"]],
                     plan=[{"name": "cancel_order", "args": {"order_id": a["id"]}},
                           {"name": "get_shipping_status", "args": {"order_id": b["id"]}}],
                     final=f"Order {a['id']} is cancelled, and {b['id']} is on its way: tracking {b['tracking']}.",
                     max_steps=10)
    raise ValueError(category)


def generate(out_dir: Path, per_category: int = 30, seed: int = 7) -> int:
    rng = random.Random(seed)
    world = build_world(rng)
    used: set[str] = set()
    tasks = [make_task(c, i, rng, world, used) for c in CATEGORIES for i in range(1, per_category + 1)]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "world.json").write_text(json.dumps({"books": world.books, "orders": world.orders}, indent=1))
    with (out_dir / "tasks.jsonl").open("w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")
    return len(tasks)


def load_suite(suite_dir: Path) -> tuple[World, list[dict[str, Any]]]:
    w = json.loads((suite_dir / "world.json").read_text())
    tasks = [json.loads(line) for line in (suite_dir / "tasks.jsonl").read_text().splitlines() if line.strip()]
    return World(books=w["books"], orders=w["orders"]), tasks
