# config.py
#
# Central settings. Paths resolve relative to this file so scripts work from any directory.
# Every value can be overridden with an environment variable.

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DB_PATH = os.environ.get("SMART_SHOP_DB", os.path.join(BASE_DIR, "db", "ecommerce.db"))
CUSTOMER_CSV = os.path.join(BASE_DIR, "customer_data_collection.csv")
PRODUCT_CSV = os.path.join(BASE_DIR, "product_recommendation_data.csv")

# 127.0.0.1, not localhost: on Windows "localhost" tries IPv6 first and Ollama only listens
# on IPv4, which adds ~2s to every call.
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
LLM_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")
EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
LLM_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", "120"))

# How many top-scored products the LLM sees when re-ranking (keeps the prompt small).
SHORTLIST_SIZE = 12
DEFAULT_TOP_K = 3
