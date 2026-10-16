import config
from utils.db import connect, init_memory_tables


def create_feedback_table():
    """Creates feedback plus the other long-term memory tables (idempotent)."""
    conn = connect(config.DB_PATH)
    init_memory_tables(conn)
    conn.close()


if __name__ == "__main__":
    create_feedback_table()
    print(f"Memory tables ready in {config.DB_PATH}")
