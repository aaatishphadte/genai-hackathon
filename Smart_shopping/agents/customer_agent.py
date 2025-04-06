# agents/customer_agent.py

import sqlite3

class CustomerAgent:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row

    def get_profile(self, customer_id):
        query = "SELECT * FROM customers WHERE customer_id = ?"
        result = self.conn.execute(query, (customer_id,)).fetchone()
        return dict(result) if result else None

    def extract_preferences(self, customer_data, llm_func):
        prompt = f"""
        The following customer has browsing history: {customer_data['browsing_history']} 
        and purchase history: {customer_data['purchase_history']}.
        Please summarize their product preferences (category, price range, style, etc.) in one sentence.
        """
        return llm_func(prompt)
