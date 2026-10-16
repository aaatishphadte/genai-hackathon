# build_embeddings.py
#
# Embeds every product with an Ollama embedding model and stores the vectors in
# product_embeddings. Once filled, RecommendationAgent adds a semantic-similarity signal
# between the customer's preference summary and each product.
#
#   ollama pull nomic-embed-text
#   python build_embeddings.py

import sys

import numpy as np

import config
from agents.customer_agent import parse_list
from utils.db import connect, init_memory_tables
from utils.ollama_interface import LLMUnavailable, ollama_embed

BATCH_SIZE = 64


def product_text(p):
    return (f"{p['category']} - {p['subcategory']} by {p['brand']}, price {p['price']:.0f}, "
            f"rating {p['product_rating']}, {p['season']} season, "
            f"{'holiday' if p['holiday'] == 'Yes' else 'everyday'} item, sold in {p['geo_location']}, "
            f"similar to {', '.join(parse_list(p['similar_products']))}")


def main():
    conn = connect(config.DB_PATH)
    init_memory_tables(conn)
    products = [dict(r) for r in conn.execute("SELECT * FROM products ORDER BY product_id")]
    conn.execute("DELETE FROM product_embeddings")

    for start in range(0, len(products), BATCH_SIZE):
        batch = products[start:start + BATCH_SIZE]
        try:
            vectors = ollama_embed([product_text(p) for p in batch])
        except LLMUnavailable as e:
            conn.rollback()
            sys.exit(f"{e}\nStart Ollama and run `ollama pull {config.EMBED_MODEL}` first.")
        conn.executemany(
            "INSERT INTO product_embeddings (product_id, model, vector) VALUES (?, ?, ?)",
            [(p["product_id"], config.EMBED_MODEL, np.asarray(v, dtype=np.float32).tobytes())
             for p, v in zip(batch, vectors)],
        )
        print(f"\rEmbedded {start + len(batch)}/{len(products)}", end="", flush=True)

    conn.commit()
    conn.close()
    print(f"\nStored {len(products)} embeddings from {config.EMBED_MODEL}.")


if __name__ == "__main__":
    main()
