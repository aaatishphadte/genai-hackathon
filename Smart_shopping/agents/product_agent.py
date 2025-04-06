# agents/product_agent.py

import sqlite3

class ProductAgent:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row

    def get_all_products(self):
        query = "SELECT * FROM products"
        return [dict(row) for row in self.conn.execute(query).fetchall()]
