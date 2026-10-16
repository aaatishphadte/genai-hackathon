import re

import numpy as np
import pytest

from agents.customer_agent import SEGMENT_STRATEGIES
from agents.recommendation_agent import RecommendationAgent
from conftest import FakeLLM


def ids(result):
    return [r["product_id"] for r in result["recommendations"]]


def test_unknown_customer_returns_error(make_system):
    rec, _ = make_system()
    assert "error" in rec.recommend("NOPE")


def test_segment_weights_sum_to_one():
    for strategy in SEGMENT_STRATEGIES.values():
        assert sum(strategy["weights"].values()) == pytest.approx(1.0)


def test_rule_based_recommends_browsed_categories_with_reasons(make_system):
    rec, _ = make_system()
    result = rec.recommend("C1", top_k=3, use_llm=False)
    assert result["llm_used"] is False
    assert len(result["recommendations"]) == 3
    assert {r["category"] for r in result["recommendations"]} <= {"Fashion", "Books"}
    assert all(r["reason"] for r in result["recommendations"])
    assert [r["rank"] for r in result["recommendations"]] == [1, 2, 3]


def test_picks_spread_across_browsed_categories(make_system):
    rec, _ = make_system()
    result = rec.recommend("C3", top_k=3, use_llm=False)
    assert {r["category"] for r in result["recommendations"]} == {"Fitness", "Fashion", "Books"}


def test_ignore_feedback_excludes_product_and_demotes_type(make_system):
    rec, feedback = make_system()
    first = rec.recommend("C2", top_k=3, use_llm=False)["recommendations"][0]
    feedback.log_feedback("C2", first["product_id"], "ignore")

    result = rec.recommend("C2", top_k=3, use_llm=False)
    assert first["product_id"] not in ids(result)
    assert first["subcategory"] in result["feedback_signals"]["disliked_subcategories"]


def test_purchase_feedback_boosts_that_product_type(make_system, conn):
    rec, feedback = make_system()
    assert rec.recommend("C2", top_k=3, use_llm=False)["recommendations"][0]["subcategory"] != "Smartwatch"
    target = conn.execute(
        "SELECT product_id, subcategory FROM products WHERE subcategory = 'Smartwatch' LIMIT 1").fetchone()
    feedback.log_feedback("C2", target["product_id"], "purchase")

    after = rec.recommend("C2", top_k=3, use_llm=False)
    assert target["product_id"] not in ids(after)
    assert after["recommendations"][0]["subcategory"] == "Smartwatch"
    assert "liked" in after["recommendations"][0]["reason"]


def test_invalid_feedback_action_rejected(make_system):
    _, feedback = make_system()
    with pytest.raises(ValueError):
        feedback.log_feedback("C1", "P000", "love")


def test_llm_choice_is_used_and_hallucinated_ids_dropped(make_system):
    rec, _ = make_system()
    shortlist = rec.product_agent.get_candidates(
        rec.customer_agent.get_profile("C1"), SEGMENT_STRATEGIES["Frequent Buyer"]["weights"])
    # Lowest-scored relevant candidate, so the LLM's choice is visible.
    chosen = [c for c in shortlist if c["signals"]["interest"] > 0][-1]["product_id"]

    llm = FakeLLM(picks=["P999", chosen, "not-an-id"])
    rec.llm_func = llm
    result = rec.recommend("C1", top_k=3)

    assert result["llm_used"] is True
    assert ids(result)[0] == chosen
    assert result["recommendations"][0]["reason"] == f"LLM reason {chosen}"
    assert "P999" not in ids(result) and len(ids(result)) == 3
    assert all(r["reason"] for r in result["recommendations"])


def test_llm_picks_still_cover_interest_categories(make_system, conn):
    rec, _ = make_system()
    fashion = [r[0] for r in conn.execute(
        "SELECT product_id FROM products WHERE category = 'Fashion' AND subcategory != 'Jeans'")]
    shortlist_ids = {c["product_id"] for c in rec.product_agent.get_candidates(
        rec.customer_agent.get_profile("C1"), SEGMENT_STRATEGIES["Frequent Buyer"]["weights"])}
    rec.llm_func = FakeLLM(picks=[p for p in fashion if p in shortlist_ids][:3])  # all one category

    result = rec.recommend("C1", top_k=3)
    assert result["llm_used"] is True
    assert {r["category"] for r in result["recommendations"]} == {"Fashion", "Books"}


def test_llm_returning_no_valid_ids_falls_back_to_rules(make_system):
    rec, _ = make_system(llm=FakeLLM(picks=["P999"]))
    result = rec.recommend("C1", top_k=3)
    assert result["llm_used"] is False
    assert len(result["recommendations"]) == 3


def test_llm_unavailable_falls_back(make_system):
    rec, _ = make_system(llm=FakeLLM(fail=True))
    result = rec.recommend("C1", top_k=3)
    assert result["llm_used"] is False
    assert len(result["recommendations"]) == 3
    assert result["preferences"].startswith("Interested in")


def test_preference_summary_cached_until_feedback_changes(make_system):
    llm = FakeLLM(picks=[])
    rec, feedback = make_system(llm=llm)
    rec.recommend("C1")
    rec.recommend("C1")
    assert llm.count("summarize their product preferences") == 1

    feedback.log_feedback("C1", "P000", "click")
    rec.recommend("C1")
    assert llm.count("summarize their product preferences") == 2


def test_llm_prompt_contains_shortlist_not_whole_catalogue(make_system, conn):
    llm = FakeLLM(picks=[])
    rec, _ = make_system(llm=llm)
    rec.recommend("C1")
    rerank_prompt = next(p for p in llm.prompts if "Candidate products" in p)
    total = conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]
    assert len(re.findall(r"- P\d+:", rerank_prompt)) <= 12 < total


def test_recommendations_logged(make_system):
    rec, feedback = make_system()
    rec.recommend("C1", top_k=3, use_llm=False)
    history = feedback.get_recommendation_history("C1")
    assert len(history) == 3 and {h["strategy"] for h in history} == {"Frequent Buyer"}


def test_semantic_signal_used_when_embeddings_present(make_system, conn):
    rows = conn.execute("SELECT product_id FROM products").fetchall()
    for i, row in enumerate(rows):
        vec = np.zeros(4, dtype=np.float32)
        vec[i % 4] = 1
        conn.execute("INSERT INTO product_embeddings VALUES (?, 'test', ?)", (row[0], vec.tobytes()))
    conn.commit()

    rec, _ = make_system(llm=FakeLLM(picks=[]), embed_func=lambda texts: [[1, 0, 0, 0]])
    result = rec.recommend("C1")
    assert all("semantic" in r["signals"] for r in result["recommendations"])


@pytest.mark.parametrize("raw, expected", [
    ('{"recommendations": [{"product_id": "P1"}]}', ["P1"]),
    ('Sure! {"recommendations": [{"product_id": "P2"}]} hope this helps', ["P2"]),
    ('[{"product_id": "P3"}]', ["P3"]),
    ("no json here", []),
    ("", []),
])
def test_parse_json_list(raw, expected):
    assert [d["product_id"] for d in RecommendationAgent._parse_json_list(raw)] == expected
