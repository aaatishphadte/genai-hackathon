# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Smart Shopping Suggestions: a multi-agent product recommender for Accenture's "Hack the Future" Gen AI hackathon (Problem Statement 2). It uses a local Ollama LLM and SQLite for long-term memory. All code lives in `Smart_shopping/`; `readme.md` explains the design and evaluation results.

## Commands

Run from `Smart_shopping/`. Paths come from `config.py` and resolve relative to that file, so the working directory doesn't matter for the DB or CSVs. Imports, however, assume `Smart_shopping/` is on `sys.path`.

```bash
pip install -r requirements.txt
python main.py [--customer C1001] [--no-llm] [--top 5] [--db path]   # CLI with feedback loop
python app.py                                  # Flask demo at http://127.0.0.1:5000
python evaluate.py --sample 0                  # leave-last-purchase-out eval vs popularity/random
pytest tests                                   # single test: pytest tests/test_agents.py::test_parse_json_list
python build_sqlite_db.py                      # rebuild db/ecommerce.db from the CSVs
python build_embeddings.py                     # optional; needs `ollama pull nomic-embed-text`
```

On this machine, `pytest` belongs to the Python 3.10 install while `python` is 3.13. If it complains about missing plugins, add `-p no:base_url -p no:playwright`.

Point `SMART_SHOP_DB` at a scratch copy when experimenting. `main.py` and `app.py` write to `recommendations`/`feedback` (and `customer_memory` when Ollama is up), so running them pollutes the committed `db/ecommerce.db`. `evaluate.py` only adds the memory tables. Other env vars: `OLLAMA_MODEL` (default `llama3`), `OLLAMA_EMBED_MODEL`, `OLLAMA_URL`, `OLLAMA_TIMEOUT`.

## Architecture

`main.build_agents()` (also used by `app.py`) shares one SQLite connection across four plain-class agents:

- `RecommendationAgent.recommend()` is the orchestrator. It gets the profile and strategy (CustomerAgent), feedback signals (FeedbackAgent) and an LLM preference summary (CustomerAgent, cached in `customer_memory` keyed by a hash of history + feedback). Then it fetches a 12-item shortlist (ProductAgent), has the LLM pick and explain top-k from that shortlist, and logs the result to `recommendations`. It returns a dict, not a string.
- `ProductAgent` loads all products into numpy arrays at init and scores the whole catalogue per request with named signals in [0, 1] (`SIGNALS`). The weights per customer segment live in `SEGMENT_STRATEGIES` in `customer_agent.py`, and each segment's weights must sum to 1 (a test checks this). The `semantic` signal turns on only when `product_embeddings` has a row for every product and an embed function is passed; otherwise its weight is dropped and the rest renormalized.
- `FeedbackAgent.get_preference_signals()` nets click=+1, purchase=+2, ignore=-1 per subcategory. Purchased and ignored product IDs are excluded from future recommendations.
- `utils/ollama_interface.py` raises `LLMUnavailable` on any failure, and every LLM call site catches it and falls back to rule-based output. Keep that contract when adding LLM calls. `llm_func` is called as `llm_func(prompt, json_mode=True)` for re-ranking.
- `utils/db.py` holds both schemas. `init_memory_tables()` is idempotent and runs on every start, so new memory tables go there.

Design decisions to preserve:

- Never put the full catalogue into a prompt; the LLM only sees the shortlist. LLM-returned product IDs are validated against the shortlist. The LLM's order is then fed back through `_rule_based_pick`, so category coverage holds even when llama3 ignores the prompt's coverage rule (it does, in practice).
- `config.OLLAMA_URL` defaults to `127.0.0.1`, not `localhost`. On Windows, `localhost` tries IPv6 first and adds ~2s to every Ollama call.
- `build_embeddings.py` takes ~10 minutes and adds ~40 MB to the DB. Don't commit an embedded `db/ecommerce.db`.
- `_rule_based_pick` spreads picks across the customer's *interest* categories first (`signals["interest"] > 0`), then across product types. `evaluate.py` calls it directly, so changes to picking show up in the eval.
- In the data, every browsed category has exactly one purchase. The "browsed but not bought" interest boost therefore does nothing on full histories but drives the eval's 100% category hit. Read the eval with that in mind.
- List columns (`browsing_history`, `purchase_history`, `similar_products`) are stored as Python-repr strings. Parse them with `agents.customer_agent.parse_list`.
