# agents/recommendation_agent.py

class RecommendationAgent:
    def __init__(self, customer_agent, product_agent, llm_func):
        self.customer_agent = customer_agent
        self.product_agent = product_agent
        self.llm_func = llm_func

    def recommend(self, customer_id):
        customer_data = self.customer_agent.get_profile(customer_id)
        if not customer_data:
            return "❌ Customer not found."

        preferences = self.customer_agent.extract_preferences(customer_data, self.llm_func)
        products = self.product_agent.get_all_products()

        prompt = f"""
        A customer prefers: {preferences}.
        Recommend 3 suitable products (just product_id and why) from the following list: 
        {products}
        """
        recommendations = self.llm_func(prompt)
        return recommendations
