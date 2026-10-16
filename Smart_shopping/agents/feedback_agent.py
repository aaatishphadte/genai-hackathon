# agents/feedback_agent.py
#
# Records how customers react to recommendations and turns that history into signals the
# other agents use on the next run (the learning loop, persisted in SQLite).

from datetime import datetime

from utils.db import connect

ACTIONS = ("click", "purchase", "ignore")


class FeedbackAgent:
    def __init__(self, db_path=None, conn=None):
        self.conn = conn or connect(db_path)

    def log_feedback(self, customer_id, product_id, action):
        if action not in ACTIONS:
            raise ValueError(f"action must be one of {ACTIONS}, got {action!r}")
        query = """
        INSERT INTO feedback (customer_id, product_id, action, timestamp)
        VALUES (?, ?, ?, ?)
        """
        self.conn.execute(query, (customer_id, product_id, action, datetime.now().isoformat(" ", timespec="seconds")))
        self.conn.commit()

    def get_feedback_summary(self, customer_id):
        query = """
        SELECT action, COUNT(*) as count
        FROM feedback
        WHERE customer_id = ?
        GROUP BY action
        """
        return {row["action"]: row["count"] for row in self.conn.execute(query, (customer_id,))}

    def get_preference_signals(self, customer_id):
        """Liked/disliked product types and products to stop recommending."""
        rows = self.conn.execute(
            """SELECT f.product_id, f.action, p.category, p.subcategory
               FROM feedback f JOIN products p ON p.product_id = f.product_id
               WHERE f.customer_id = ? ORDER BY f.id""",
            (customer_id,),
        ).fetchall()

        # Net score per subcategory: purchases count most, ignores count against.
        weight = {"purchase": 2, "click": 1, "ignore": -1}
        sub_score, sub_category, excluded = {}, {}, set()
        for row in rows:
            sub_score[row["subcategory"]] = sub_score.get(row["subcategory"], 0) + weight[row["action"]]
            sub_category[row["subcategory"]] = row["category"]
            if row["action"] in ("purchase", "ignore"):
                excluded.add(row["product_id"])

        liked = {s for s, v in sub_score.items() if v > 0}
        return {
            "liked_subcategories": liked,
            "liked_categories": {sub_category[s] for s in liked},
            "disliked_subcategories": {s for s, v in sub_score.items() if v < 0},
            "excluded_products": excluded,
        }

    def get_recommendation_history(self, customer_id, limit=20):
        query = """
        SELECT r.timestamp, r.product_id, r.rank, r.score, r.strategy, f.action
        FROM recommendations r
        LEFT JOIN feedback f ON f.customer_id = r.customer_id AND f.product_id = r.product_id
        WHERE r.customer_id = ?
        ORDER BY r.id DESC LIMIT ?
        """
        return [dict(row) for row in self.conn.execute(query, (customer_id, limit))]
