import sqlite3

def create_feedback_table():
    conn = sqlite3.connect("db/ecommerce.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        customer_id TEXT NOT NULL,
        product_id TEXT NOT NULL,
        action TEXT CHECK(action IN ('click', 'purchase', 'ignore')) NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    conn.commit()
    conn.close()

create_feedback_table()
