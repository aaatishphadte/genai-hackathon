# main.py

from agents.customer_agent import CustomerAgent
from agents.product_agent import ProductAgent
from agents.recommendation_agent import RecommendationAgent
from utils.ollama_interface import ollama_llm

def main():
    db_path = "db/ecommerce.db"  # Make sure the DB is in this folder
    customer_id = input("Enter Customer ID: ").strip()

    customer_agent = CustomerAgent(db_path)
    product_agent = ProductAgent(db_path)
    rec_agent = RecommendationAgent(customer_agent, product_agent, ollama_llm)

    print("\n📦 Recommended Products:\n")
    recommendations = rec_agent.recommend(customer_id)
    print(recommendations)

if __name__ == "__main__":
    main()
