# agents/customer_agent.py
#
# Understands the customer: profile, segment strategy, and a natural-language preference
# summary that is cached in SQLite (customer_memory) as long-term memory.

import ast
import hashlib

from utils.db import connect
from utils.ollama_interface import LLMUnavailable

# Scoring weights per customer segment (see ProductAgent.score_products for each signal).
# - New visitors have little history, so lean on popularity and quality ("most popular items").
# - Frequent buyers get items related to what they already bought.
# - Occasional shoppers get a balance plus seasonal/holiday picks to re-engage them.
SEGMENT_STRATEGIES = {
    "New Visitor": {
        "description": "popular, highly rated items in the categories they browsed",
        "weights": {"interest": 0.25, "related": 0.05, "popularity": 0.30, "quality": 0.25,
                    "seasonal": 0.05, "price_fit": 0.05, "semantic": 0.05},
    },
    "Frequent Buyer": {
        "description": "items related to and complementing their past purchases",
        "weights": {"interest": 0.20, "related": 0.35, "popularity": 0.10, "quality": 0.15,
                    "seasonal": 0.05, "price_fit": 0.10, "semantic": 0.05},
    },
    "Occasional Shopper": {
        "description": "seasonal and holiday picks matched to their interests and budget",
        "weights": {"interest": 0.25, "related": 0.15, "popularity": 0.15, "quality": 0.10,
                    "seasonal": 0.20, "price_fit": 0.10, "semantic": 0.05},
    },
}
DEFAULT_SEGMENT = "Occasional Shopper"


def parse_list(value):
    if isinstance(value, list):
        return value
    try:
        parsed = ast.literal_eval(value) if value else []
        return list(parsed) if isinstance(parsed, (list, tuple)) else [parsed]
    except (ValueError, SyntaxError):
        return [v.strip() for v in str(value).split(",") if v.strip()]


class CustomerAgent:
    def __init__(self, db_path=None, conn=None):
        self.conn = conn or connect(db_path)

    def get_profile(self, customer_id):
        query = "SELECT * FROM customers WHERE customer_id = ?"
        result = self.conn.execute(query, (customer_id,)).fetchone()
        if not result:
            return None
        profile = dict(result)
        profile["browsing_history"] = parse_list(profile["browsing_history"])
        profile["purchase_history"] = parse_list(profile["purchase_history"])
        return profile

    def list_customer_ids(self, limit=None):
        query = "SELECT customer_id FROM customers ORDER BY customer_id"
        if limit:
            query += f" LIMIT {int(limit)}"
        return [row[0] for row in self.conn.execute(query)]

    def get_strategy(self, customer_data):
        segment = customer_data.get("customer_segment")
        name = segment if segment in SEGMENT_STRATEGIES else DEFAULT_SEGMENT
        return {"segment": name, **SEGMENT_STRATEGIES[name]}

    def extract_preferences(self, customer_data, llm_func, feedback_signals=None):
        """One-sentence preference summary. Cached until history or feedback changes."""
        feedback_text = self._describe_feedback(feedback_signals)
        source_hash = hashlib.sha1(
            repr((customer_data["browsing_history"], customer_data["purchase_history"],
                  customer_data["avg_order_value"], feedback_text)).encode()
        ).hexdigest()

        cached = self.conn.execute(
            "SELECT preference_summary, source_hash FROM customer_memory WHERE customer_id = ?",
            (customer_data["customer_id"],),
        ).fetchone()
        if cached and cached["source_hash"] == source_hash:
            return cached["preference_summary"]

        prompt = f"""
        The following customer has browsing history: {customer_data['browsing_history']}
        and purchase history: {customer_data['purchase_history']}.
        Their average order value is Rs.{customer_data['avg_order_value']:.0f} (Indian rupees), they are a
        "{customer_data['customer_segment']}" and it is currently {customer_data['season']}.
        {feedback_text}
        Please summarize their product preferences (category, price range, style, etc.) in one sentence.
        """
        if llm_func is None:
            return self._rule_based_summary(customer_data, feedback_text)
        try:
            summary = llm_func(prompt).strip()
        except LLMUnavailable:
            # Don't cache the fallback, so the LLM summary is generated once Ollama is up.
            return self._rule_based_summary(customer_data, feedback_text)
        if not summary:
            return self._rule_based_summary(customer_data, feedback_text)

        self.conn.execute(
            """INSERT INTO customer_memory (customer_id, preference_summary, source_hash, updated_at)
               VALUES (?, ?, ?, CURRENT_TIMESTAMP)
               ON CONFLICT(customer_id) DO UPDATE SET preference_summary = excluded.preference_summary,
                   source_hash = excluded.source_hash, updated_at = CURRENT_TIMESTAMP""",
            (customer_data["customer_id"], summary, source_hash),
        )
        self.conn.commit()
        return summary

    @staticmethod
    def _describe_feedback(signals):
        if not signals:
            return ""
        parts = []
        if signals.get("liked_subcategories"):
            parts.append(f"Recently engaged with: {', '.join(sorted(signals['liked_subcategories']))}.")
        if signals.get("disliked_subcategories"):
            parts.append(f"Recently ignored: {', '.join(sorted(signals['disliked_subcategories']))}.")
        return " ".join(parts)

    @staticmethod
    def _rule_based_summary(customer_data, feedback_text=""):
        interests = ", ".join(customer_data["browsing_history"]) or "general products"
        bought = ", ".join(customer_data["purchase_history"]) or "nothing yet"
        aov = customer_data["avg_order_value"]
        summary = (f"Interested in {interests}; previously bought {bought}; "
                   f"typical spend around {aov:.0f} per order.")
        return f"{summary} {feedback_text}".strip()
