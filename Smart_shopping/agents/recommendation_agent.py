# agents/recommendation_agent.py
#
# Orchestrator. Pipeline per request:
#   1. CustomerAgent  -> profile + segment strategy
#   2. FeedbackAgent  -> learned likes/dislikes from earlier sessions
#   3. CustomerAgent  -> preference summary (LLM, cached in long-term memory)
#   4. ProductAgent   -> score the catalogue, return a short diverse candidate list
#   5. LLM            -> pick the final items from the shortlist and explain why
#   6. Log what was served to the recommendations table

import json
import re
from datetime import datetime

import config
from utils.ollama_interface import LLMUnavailable

SIGNAL_PHRASES = {
    "interest": "matches the {category} items you've been browsing",
    "related": "goes well with your earlier purchases",
    "popularity": "popular with similar shoppers",
    "quality": "highly rated with positive reviews",
    "seasonal": "a good pick for {season}",
    "price_fit": "fits your usual budget",
    "semantic": "closely matches your style profile",
}


class RecommendationAgent:
    def __init__(self, customer_agent, product_agent, llm_func, feedback_agent=None, embed_func=None):
        self.customer_agent = customer_agent
        self.product_agent = product_agent
        self.llm_func = llm_func
        self.feedback_agent = feedback_agent
        self.embed_func = embed_func

    def recommend(self, customer_id, top_k=config.DEFAULT_TOP_K, use_llm=True, log=True):
        customer_data = self.customer_agent.get_profile(customer_id)
        if not customer_data:
            return {"error": f"Customer {customer_id!r} not found."}

        strategy = self.customer_agent.get_strategy(customer_data)
        signals = self.feedback_agent.get_preference_signals(customer_id) if self.feedback_agent else {}
        llm = self.llm_func if use_llm else None
        preferences = self.customer_agent.extract_preferences(customer_data, llm, signals)

        query_vector = self._embed(preferences) if use_llm else None
        candidates = self.product_agent.get_candidates(
            customer_data, strategy["weights"], signals, query_vector,
            limit=max(config.SHORTLIST_SIZE, top_k),
        )

        picks, llm_used = None, False
        if llm:
            picks = self._llm_rerank(customer_data, strategy, preferences, candidates, top_k)
            llm_used = picks is not None
        if not picks:
            picks = self._rule_based_pick(candidates, top_k)

        for rank, item in enumerate(picks, start=1):
            item["rank"] = rank
            if not item.get("reason"):
                item["reason"] = self._explain(item, strategy["weights"], customer_data, signals)

        if log:
            self._log(customer_id, picks, strategy["segment"])

        return {
            "customer": customer_data,
            "strategy": strategy,
            "preferences": preferences,
            "feedback_signals": signals,
            "recommendations": picks,
            "llm_used": llm_used,
        }

    def _embed(self, text):
        if not self.embed_func or self.product_agent.embeddings is None:
            return None
        try:
            return self.embed_func([text])[0]
        except LLMUnavailable:
            return None

    def _llm_rerank(self, customer_data, strategy, preferences, candidates, top_k):
        lines = "\n".join(
            f"- {c['product_id']}: {c['category']} / {c['subcategory']}, {c['brand']}, "
            f"price Rs.{c['price']:.0f}, rating {c['product_rating']}/5, season {c['season']}, "
            f"holiday item {c['holiday']}, match score {c['score']:.2f}"
            f"{' [matches interests]' if c['signals'].get('interest', 0) > 0 else ''}"
            for c in candidates
        )
        interests = sorted({c["category"] for c in candidates if c["signals"].get("interest", 0) > 0})
        prompt = f"""
        You are a shopping assistant for an Indian e-commerce store (prices in rupees).
        Customer segment: {strategy['segment']} -> focus on {strategy['description']}.
        Customer preferences: {preferences}
        Categories the customer is interested in: {', '.join(interests) or 'none yet'}.
        Current season: {customer_data['season']}. Shopping during a holiday: {customer_data['holiday']}.

        Candidate products (pre-ranked by match score):
        {lines}

        Choose the {top_k} best products for this customer from the candidates only.
        Rules:
        - Pick products marked [matches interests]; only pick others if fewer than {top_k} are marked.
        - Cover as many different interest categories as possible, then different product types.
        - Prefer higher match scores when otherwise equal.
        For each, give a short, customer-facing reason (max 20 words) that cites a concrete detail
        listed above (price, rating, season, holiday, or link to their history). Do not invent
        product features that are not listed.
        Respond with JSON only, in this format:
        {{"recommendations": [{{"product_id": "P1234", "reason": "..."}}]}}
        """
        try:
            raw = self.llm_func(prompt, json_mode=True)
        except LLMUnavailable:
            return None
        except TypeError:
            # llm_func without json_mode support (e.g. a custom backend).
            try:
                raw = self.llm_func(prompt)
            except LLMUnavailable:
                return None

        by_id = {c["product_id"]: c for c in candidates}
        llm_picks, seen = [], set()
        for entry in self._parse_json_list(raw):
            pid = str(entry.get("product_id", "")).strip()
            if pid in by_id and pid not in seen:
                seen.add(pid)
                llm_picks.append({**by_id[pid], "reason": str(entry.get("reason", "")).strip()})
        if not llm_picks:
            return None
        # The LLM's order is a preference; the same coverage rules as the rule-based path
        # still apply (small models often ignore "cover every interest category").
        # Remaining slots are filled from the score-ranked candidates.
        ordered = llm_picks + [c for c in candidates if c["product_id"] not in seen]
        return self._rule_based_pick(ordered, top_k)

    @staticmethod
    def _parse_json_list(raw):
        if not raw:
            return []
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                return []
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                return []
        if isinstance(data, dict):
            data = data.get("recommendations", [])
        return [d for d in data if isinstance(d, dict)] if isinstance(data, list) else []

    @staticmethod
    def _rule_based_pick(candidates, top_k):
        """Highest scores, spread across the customer's interest categories first, then
        across product types, then fill by score."""
        picks, chosen = [], set()
        relevant = lambda c: c["signals"].get("interest", 0) > 0  # noqa: E731
        rules = (
            lambda c: relevant(c) and c["category"] not in {p["category"] for p in picks},
            lambda c: relevant(c) and c["subcategory"] not in {p["subcategory"] for p in picks},
            lambda c: True,
        )
        for allowed in rules:
            for c in candidates:
                if len(picks) == top_k:
                    return picks
                if c["product_id"] not in chosen and allowed(c):
                    picks.append(dict(c))
                    chosen.add(c["product_id"])
        return picks

    @staticmethod
    def _explain(item, weights, customer_data, feedback_signals=None):
        contributions = sorted(
            ((weights.get(name, 0) * value, name) for name, value in item["signals"].items()),
            reverse=True,
        )
        liked = item["subcategory"] in (feedback_signals or {}).get("liked_subcategories", ())
        phrases = [
            f"similar to the {item['subcategory']} items you liked" if name == "interest" and liked
            else SIGNAL_PHRASES[name].format(category=item["category"], season=customer_data["season"])
            for contribution, name in contributions[:2] if contribution > 0
        ]
        text = " and ".join(phrases) or "a strong overall match"
        return text[0].upper() + text[1:] + "."

    def _log(self, customer_id, picks, segment):
        conn = self.customer_agent.conn
        now = datetime.now().isoformat(" ", timespec="seconds")  # local time, like FeedbackAgent
        conn.executemany(
            """INSERT INTO recommendations (customer_id, product_id, rank, score, reason, strategy, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            [(customer_id, p["product_id"], p["rank"], p["score"], p["reason"], segment, now) for p in picks],
        )
        conn.commit()
