# evaluate.py
#
# Offline evaluation (no LLM needed). For each sampled customer, hide their last purchase,
# recommend top-k from the remaining history, and check whether the hidden item's
# subcategory / category was recovered. Compared against popularity and random baselines.
#
#   python evaluate.py --sample 2000 --k 3

import argparse
import random

from agents.customer_agent import CustomerAgent
from agents.product_agent import ProductAgent
from agents.recommendation_agent import RecommendationAgent
from utils.db import connect, init_memory_tables


def hit_rates(recommended, held_out_sub, sub_to_cat):
    subs = {p["subcategory"] for p in recommended}
    cats = {p["category"] for p in recommended}
    return held_out_sub in subs, sub_to_cat[held_out_sub] in cats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=2000, help="customers to evaluate (0 = all)")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--db")
    args = parser.parse_args()

    conn = connect(args.db)
    init_memory_tables(conn)
    customers = CustomerAgent(conn=conn)
    products = ProductAgent(conn=conn)
    rng = random.Random(args.seed)

    ids = customers.list_customer_ids()
    if args.sample:
        ids = rng.sample(ids, min(args.sample, len(ids)))

    popular = products.popular_products(limit=200)
    methods = ("multi-agent", "popularity", "random")
    totals = {m: {"sub": 0, "cat": 0} for m in methods}
    per_segment = {}

    for cid in ids:
        profile = customers.get_profile(cid)
        held_out = profile["purchase_history"][-1]
        profile = {**profile, "purchase_history": profile["purchase_history"][:-1]}
        strategy = customers.get_strategy(profile)

        recs = {
            "multi-agent": RecommendationAgent._rule_based_pick(
                products.get_candidates(profile, strategy["weights"], limit=12), args.k),
            # One per subcategory, so the baseline gets the same diversity advantage.
            "popularity": list({p["subcategory"]: p for p in reversed(popular)}.values())[::-1][: args.k],
            "random": rng.sample(products.products, args.k),
        }
        seg = per_segment.setdefault(strategy["segment"], {"n": 0, "sub": 0, "cat": 0})
        seg["n"] += 1
        for method, items in recs.items():
            sub_hit, cat_hit = hit_rates(items, held_out, products.subcategory_to_category)
            totals[method]["sub"] += sub_hit
            totals[method]["cat"] += cat_hit
            if method == "multi-agent":
                seg["sub"] += sub_hit
                seg["cat"] += cat_hit

    n = len(ids)
    print(f"Leave-last-purchase-out evaluation on {n} customers, top-{args.k}\n")
    print(f"{'method':<14}{'subcategory hit@k':>20}{'category hit@k':>18}")
    for method in methods:
        print(f"{method:<14}{totals[method]['sub'] / n:>20.1%}{totals[method]['cat'] / n:>18.1%}")
    print("\nMulti-agent by segment:")
    for name, s in sorted(per_segment.items()):
        print(f"  {name:<20} n={s['n']:<5} subcategory {s['sub'] / s['n']:.1%}  category {s['cat'] / s['n']:.1%}")


if __name__ == "__main__":
    main()
