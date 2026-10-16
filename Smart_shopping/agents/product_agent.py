# agents/product_agent.py
#
# Knows the catalogue. Scores every product against a customer with weighted signals and
# returns a small, diverse candidate list (the retrieval step before the LLM re-ranks).

import numpy as np

from agents.customer_agent import parse_list
from utils.db import connect

SIGNALS = ("interest", "related", "popularity", "quality", "seasonal", "price_fit", "semantic")

# Customer locations are Indian cities; products carry a country.
CITY_COUNTRY = {"Kolkata": "India", "Bangalore": "India", "Delhi": "India",
                "Chennai": "India", "Mumbai": "India"}


class ProductAgent:
    def __init__(self, db_path=None, conn=None):
        self.conn = conn or connect(db_path)
        self.products = self.get_all_products()
        self.index = {p["product_id"]: i for i, p in enumerate(self.products)}
        for p in self.products:
            p["similar_products"] = parse_list(p["similar_products"])
        self.subcategory_to_category = {p["subcategory"]: p["category"] for p in self.products}
        self._build_arrays()
        self._load_embeddings()

    def get_all_products(self):
        query = "SELECT * FROM products ORDER BY product_id"
        return [dict(row) for row in self.conn.execute(query).fetchall()]

    def get_product(self, product_id):
        i = self.index.get(product_id)
        return self.products[i] if i is not None else None

    def _build_arrays(self):
        P = self.products
        self.category = np.array([p["category"] for p in P])
        self.subcategory = np.array([p["subcategory"] for p in P])
        self.price = np.array([p["price"] for p in P], dtype=float)
        self.season = np.array([p["season"] for p in P])
        self.holiday = np.array([p["holiday"] for p in P])
        self.geo = np.array([p["geo_location"] for p in P])
        self.popularity = np.array([p["probability_of_recommendation"] for p in P], dtype=float)
        rating = np.array([p["product_rating"] for p in P], dtype=float)
        similar_rating = np.array([p["avg_rating_similar"] for p in P], dtype=float)
        sentiment = np.array([p["sentiment_score"] for p in P], dtype=float)
        self.quality = 0.5 * rating / 5 + 0.3 * sentiment + 0.2 * similar_rating / 5

        # Multi-hot matrix: product x subcategory it is listed as similar to.
        self.subcategories = sorted(self.subcategory_to_category)
        sub_idx = {s: j for j, s in enumerate(self.subcategories)}
        self.similar_matrix = np.zeros((len(P), len(self.subcategories)), dtype=bool)
        for i, p in enumerate(P):
            for s in p["similar_products"]:
                if s in sub_idx:
                    self.similar_matrix[i, sub_idx[s]] = True
        self._sub_idx = sub_idx

    def _load_embeddings(self):
        self.embeddings = None
        rows = self.conn.execute("SELECT product_id, vector FROM product_embeddings").fetchall() \
            if self._table_exists("product_embeddings") else []
        if len(rows) != len(self.products):
            return
        dim = len(rows[0]["vector"]) // 4
        matrix = np.zeros((len(self.products), dim), dtype=np.float32)
        for row in rows:
            matrix[self.index[row["product_id"]]] = np.frombuffer(row["vector"], dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        self.embeddings = matrix / np.where(norms == 0, 1, norms)

    def _table_exists(self, name):
        return self.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone() is not None

    def score_products(self, customer, weights, feedback_signals=None, query_vector=None):
        """Return (total_score, {signal: array}) for every product, each signal in [0, 1]."""
        fb = feedback_signals or {}
        browsed = customer["browsing_history"]
        purchased = customer["purchase_history"]
        purchased_categories = {self.subcategory_to_category.get(s) for s in purchased}

        # Browsed but never bought from = open purchase intent (browse-abandonment retargeting).
        unbought = [c for c in browsed if c not in purchased_categories]
        interest = np.where(np.isin(self.category, browsed), 0.6, 0.0)
        interest = np.where(np.isin(self.category, unbought), 1.0, interest)
        interest = np.maximum(interest, np.where(np.isin(self.category, list(fb.get("liked_categories", []))), 0.8, 0))
        interest = np.maximum(interest, np.where(np.isin(self.subcategory, list(fb.get("liked_subcategories", []))), 1.0, 0))
        interest = np.where(np.isin(self.subcategory, list(fb.get("disliked_subcategories", []))), interest * 0.3, interest)

        # Related to past purchases: complementary (lists a purchased item as similar),
        # same product type (repeat), or same category, different type.
        purchased_cols = [self._sub_idx[s] for s in purchased if s in self._sub_idx]
        complementary = self.similar_matrix[:, purchased_cols].any(axis=1) if purchased_cols \
            else np.zeros(len(self.products), dtype=bool)
        related = np.where(complementary, 1.0, 0.0)
        related = np.maximum(related, np.where(np.isin(self.subcategory, purchased), 0.6, 0))
        related = np.maximum(related, np.where(np.isin(self.category, list(purchased_categories)), 0.5, 0))

        country = CITY_COUNTRY.get(customer.get("location"), "India")
        seasonal = (0.5 * (self.season == customer.get("season"))
                    + 0.3 * (self.holiday == customer.get("holiday"))
                    + 0.2 * (self.geo == country))

        aov = max(float(customer.get("avg_order_value") or 0), 1.0)
        price_fit = np.clip(1 - np.abs(self.price - aov) / aov, 0, 1)

        signals = {
            "interest": interest,
            "related": related,
            "popularity": self.popularity,
            "quality": self.quality,
            "seasonal": seasonal,
            "price_fit": price_fit,
        }
        active = dict(weights)
        if query_vector is not None and self.embeddings is not None:
            q = np.asarray(query_vector, dtype=np.float32)
            q = q / (np.linalg.norm(q) or 1)
            signals["semantic"] = (self.embeddings @ q + 1) / 2
        else:
            active.pop("semantic", None)

        total_weight = sum(active.values()) or 1
        score = sum(active[name] * signals[name] for name in active) / total_weight
        return score, signals

    def get_candidates(self, customer, weights, feedback_signals=None, query_vector=None,
                       limit=12, per_subcategory=2, per_category=None):
        """Top products by score, skipping excluded ones and capping repeats so the
        shortlist spans several product types and categories."""
        fb = feedback_signals or {}
        score, signals = self.score_products(customer, weights, fb, query_vector)
        excluded = set(fb.get("excluded_products", []))
        if per_category is None:
            # Share the shortlist between the categories the customer cares about, so a
            # one-category customer isn't padded out with unrelated categories.
            interests = set(customer["browsing_history"]) | set(fb.get("liked_categories", ()))
            per_category = max(4, limit // max(1, len(interests)))

        candidates, per_sub, per_cat = [], {}, {}
        for i in np.argsort(-score, kind="stable"):
            product = self.products[i]
            if product["product_id"] in excluded:
                continue
            sub, cat = product["subcategory"], product["category"]
            if per_sub.get(sub, 0) >= per_subcategory or per_cat.get(cat, 0) >= per_category:
                continue
            per_sub[sub] = per_sub.get(sub, 0) + 1
            per_cat[cat] = per_cat.get(cat, 0) + 1
            candidates.append({
                **product,
                "score": round(float(score[i]), 4),
                "signals": {name: round(float(values[i]), 3) for name, values in signals.items()},
            })
            if len(candidates) >= limit:
                break
        return candidates

    def popular_products(self, limit=3, exclude=()):
        """Non-personalized baseline: highest probability_of_recommendation."""
        order = np.argsort(-self.popularity, kind="stable")
        picks = [self.products[i] for i in order if self.products[i]["product_id"] not in exclude]
        return picks[:limit]
