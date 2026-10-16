# main.py
#
# CLI: recommend products for a customer, collect feedback, and recommend again.
#   python main.py                      interactive
#   python main.py --customer C1000     one customer
#   python main.py --no-llm             rule-based only (no Ollama needed)

import argparse

import config
from agents.customer_agent import CustomerAgent
from agents.feedback_agent import ACTIONS, FeedbackAgent
from agents.product_agent import ProductAgent
from agents.recommendation_agent import RecommendationAgent
from utils.db import connect, init_memory_tables
from utils.ollama_interface import is_ollama_available, ollama_embed, ollama_llm


def build_agents(db_path=None):
    conn = connect(db_path)
    init_memory_tables(conn)
    customer_agent = CustomerAgent(conn=conn)
    product_agent = ProductAgent(conn=conn)
    feedback_agent = FeedbackAgent(conn=conn)
    rec_agent = RecommendationAgent(customer_agent, product_agent, ollama_llm,
                                    feedback_agent=feedback_agent, embed_func=ollama_embed)
    return rec_agent, feedback_agent


def print_result(result):
    if "error" in result:
        print(f"\n❌ {result['error']}")
        return
    c = result["customer"]
    print(f"\n👤 {c['customer_id']} | {c['age']} y/o {c['gender']}, {c['location']} | "
          f"{c['customer_segment']} | avg order {c['avg_order_value']:.0f}")
    print(f"   Browsed: {', '.join(c['browsing_history'])} | Bought: {', '.join(c['purchase_history'])}")
    print(f"🎯 Strategy: {result['strategy']['description']}")
    print(f"🧠 Preferences: {result['preferences']}")
    source = "LLM re-ranked" if result["llm_used"] else "rule-based (LLM offline)"
    print(f"\n📦 Recommended Products ({source}):\n")
    for item in result["recommendations"]:
        print(f"  {item['rank']}. {item['product_id']}  {item['category']} / {item['subcategory']}"
              f"  ({item['brand']}, ₹{item['price']:.0f}, ★{item['product_rating']})  score {item['score']:.2f}")
        print(f"     ↳ {item['reason']}")


def collect_feedback(feedback_agent, result):
    """Returns True if any feedback was recorded."""
    items = {str(i["rank"]): i for i in result["recommendations"]}
    print("\nFeedback: e.g. '1 purchase', '2 click', '3 ignore' (several separated by commas). Enter to skip.")
    raw = input("> ").strip()
    recorded = False
    for part in filter(None, (p.strip() for p in raw.split(","))):
        tokens = part.split()
        if len(tokens) != 2 or tokens[0] not in items or tokens[1] not in ACTIONS:
            print(f"  Skipped {part!r}: use '<rank> <{'|'.join(ACTIONS)}>'")
            continue
        item = items[tokens[0]]
        feedback_agent.log_feedback(result["customer"]["customer_id"], item["product_id"], tokens[1])
        print(f"  Logged {tokens[1]} for {item['product_id']}")
        recorded = True
    return recorded


def main():
    parser = argparse.ArgumentParser(description="Smart Shopping multi-agent recommender")
    parser.add_argument("--customer", help="Customer ID, e.g. C1000")
    parser.add_argument("--top", type=int, default=config.DEFAULT_TOP_K, help="number of recommendations")
    parser.add_argument("--no-llm", action="store_true", help="skip Ollama and use rule-based ranking")
    parser.add_argument("--db", help="path to the SQLite database")
    args = parser.parse_args()

    use_llm = not args.no_llm and is_ollama_available()
    if not args.no_llm and not use_llm:
        print(f"⚠️  Ollama not reachable at {config.OLLAMA_URL}; using rule-based ranking and explanations.")

    rec_agent, feedback_agent = build_agents(args.db)
    customer_id = args.customer or input("Enter Customer ID (e.g. C1000): ").strip()

    while customer_id:
        result = rec_agent.recommend(customer_id.upper(), top_k=args.top, use_llm=use_llm)
        print_result(result)
        if "error" not in result and collect_feedback(feedback_agent, result):
            print("\n🔁 Updated recommendations using your feedback:")
            print_result(rec_agent.recommend(customer_id.upper(), top_k=args.top, use_llm=use_llm))
        if args.customer:
            break
        customer_id = input("\nEnter another Customer ID (Enter to quit): ").strip()


if __name__ == "__main__":
    main()
