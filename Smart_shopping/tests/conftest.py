import itertools
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.customer_agent import CustomerAgent  # noqa: E402
from agents.feedback_agent import FeedbackAgent  # noqa: E402
from agents.product_agent import ProductAgent  # noqa: E402
from agents.recommendation_agent import RecommendationAgent  # noqa: E402
from utils.db import connect, init_catalog_tables, init_memory_tables  # noqa: E402
from utils.ollama_interface import LLMUnavailable  # noqa: E402

CATALOG = {
    "Fashion": ["Jeans", "T-shirt", "Shoes", "Jacket"],
    "Books": ["Fiction", "Biography", "Comics", "Non-fiction"],
    "Electronics": ["Laptop", "Smartphone", "Headphones", "Smartwatch"],
    "Fitness": ["Yoga Mat", "Dumbbells", "Treadmill", "Resistance Bands"],
}

CUSTOMERS = [
    # id, segment, browsed, purchased, avg order value
    ("C1", "Frequent Buyer", ["Fashion", "Books"], ["Jeans", "Biography"], 1000.0),
    ("C2", "New Visitor", ["Electronics"], ["Laptop"], 3000.0),
    ("C3", "Occasional Shopper", ["Fitness", "Fashion", "Books"], ["Yoga Mat", "Shoes"], 1500.0),
]


@pytest.fixture
def conn(tmp_path):
    conn = connect(str(tmp_path / "test.db"))
    init_catalog_tables(conn)
    init_memory_tables(conn)
    for cid, segment, browsed, purchased, aov in CUSTOMERS:
        conn.execute("INSERT INTO customers VALUES (?,?,?,?,?,?,?,?,?,?)",
                     (cid, 30, "Female", "Delhi", str(browsed), str(purchased), segment, aov, "No", "Winter"))
    n = itertools.count()
    for category, subs in CATALOG.items():
        for sub in subs:
            for variant in range(3):
                i = next(n)
                conn.execute(
                    "INSERT INTO products VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (f"P{i:03d}", category, sub, 500.0 + 250 * variant + 37 * (i % 7), f"Brand {'ABC'[variant]}",
                     4.0, 3.0 + variant * 0.8, 0.3 + 0.2 * variant, "No" if variant else "Yes",
                     ["Winter", "Summer", "Autumn"][variant], "India", str([sub, subs[(subs.index(sub) + 1) % 4]]),
                     round(0.3 + 0.2 * variant + 0.01 * (i % 5), 2)),
                )
    conn.commit()
    return conn


class FakeLLM:
    """Records prompts; answers summaries with text and re-ranking with JSON."""

    def __init__(self, picks=None, fail=False):
        self.prompts = []
        self.picks = picks
        self.fail = fail

    def __call__(self, prompt, json_mode=False):
        self.prompts.append(prompt)
        if self.fail:
            raise LLMUnavailable("offline")
        if not json_mode:
            return "Likes affordable fashion and books."
        picks = self.picks if self.picks is not None else []
        return json.dumps({"recommendations": [{"product_id": p, "reason": f"LLM reason {p}"} for p in picks]})

    def count(self, needle):
        return sum(needle in p for p in self.prompts)


@pytest.fixture
def make_system(conn):
    def build(llm=None, embed_func=None):
        customer_agent = CustomerAgent(conn=conn)
        product_agent = ProductAgent(conn=conn)
        feedback_agent = FeedbackAgent(conn=conn)
        rec = RecommendationAgent(customer_agent, product_agent, llm or FakeLLM(),
                                  feedback_agent=feedback_agent, embed_func=embed_func)
        return rec, feedback_agent
    return build
