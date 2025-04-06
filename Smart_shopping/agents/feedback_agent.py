# agents/feedback_agent.py

import sqlite3
from datetime import datetime

class FeedbackAgent:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)

    def log_feedback(self, customer_id, product_id, action):
        query = """
        INSERT INTO feedback (customer_id, product_id, action, timestamp)
        VALUES (?, ?, ?, ?)
        """
        self.conn.execute(query, (customer_id, product_id, action, datetime.now()))
        self.conn.commit()

    def get_feedback_summary(self, customer_id):
        query = """
        SELECT action, COUNT(*) as count
        FROM feedback
        WHERE customer_id = ?
        GROUP BY action
        """
        return self.conn.execute(query, (customer_id,)).fetchall()
