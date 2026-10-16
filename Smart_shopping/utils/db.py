# utils/db.py
#
# SQLite access and the long-term memory schema shared by all agents.

import sqlite3

import config

CATALOG_SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (
    customer_id TEXT PRIMARY KEY,
    age INTEGER,
    gender TEXT,
    location TEXT,
    browsing_history TEXT,
    purchase_history TEXT,
    customer_segment TEXT,
    avg_order_value REAL,
    holiday TEXT,
    season TEXT
);
CREATE TABLE IF NOT EXISTS products (
    product_id TEXT PRIMARY KEY,
    category TEXT,
    subcategory TEXT,
    price REAL,
    brand TEXT,
    avg_rating_similar REAL,
    product_rating REAL,
    sentiment_score REAL,
    holiday TEXT,
    season TEXT,
    geo_location TEXT,
    similar_products TEXT,
    probability_of_recommendation REAL
);
"""

MEMORY_SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id TEXT NOT NULL,
    product_id TEXT NOT NULL,
    action TEXT CHECK(action IN ('click', 'purchase', 'ignore')) NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- LLM-generated preference summaries, reused until the customer's history or feedback changes.
CREATE TABLE IF NOT EXISTS customer_memory (
    customer_id TEXT PRIMARY KEY,
    preference_summary TEXT NOT NULL,
    source_hash TEXT NOT NULL,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Every recommendation served, so later runs can avoid repeats and feedback can be traced.
CREATE TABLE IF NOT EXISTS recommendations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id TEXT NOT NULL,
    product_id TEXT NOT NULL,
    rank INTEGER NOT NULL,
    score REAL NOT NULL,
    reason TEXT,
    strategy TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Optional Ollama embeddings (float32 bytes) for semantic retrieval; filled by build_embeddings.py.
CREATE TABLE IF NOT EXISTS product_embeddings (
    product_id TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    vector BLOB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_feedback_customer ON feedback(customer_id);
CREATE INDEX IF NOT EXISTS idx_recommendations_customer ON recommendations(customer_id);
"""


def connect(db_path=None):
    conn = sqlite3.connect(db_path or config.DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_catalog_tables(conn):
    conn.executescript(CATALOG_SCHEMA)
    conn.commit()


def init_memory_tables(conn):
    conn.executescript(MEMORY_SCHEMA)
    conn.commit()
