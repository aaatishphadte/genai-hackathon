# app.py
#
# Web demo: pick a customer, see the agents' recommendations, give feedback, watch them adapt.
#   python app.py   ->  http://127.0.0.1:5000

import os
import threading

from flask import Flask, jsonify, redirect, render_template, request, url_for

import config
from agents.feedback_agent import ACTIONS
from main import build_agents
from utils.ollama_interface import is_ollama_available

app = Flask(__name__)
rec_agent, feedback_agent = build_agents()
lock = threading.Lock()  # one shared SQLite connection


def sample_customers(per_segment=4):
    rows = rec_agent.customer_agent.conn.execute(
        """SELECT customer_segment, customer_id FROM customers
           WHERE customer_id IN (SELECT MIN(customer_id) FROM customers GROUP BY customer_segment)
              OR rowid % 997 = 0
           ORDER BY customer_segment, customer_id"""
    ).fetchall()
    samples = {}
    for row in rows:
        ids = samples.setdefault(row["customer_segment"], [])
        if len(ids) < per_segment:
            ids.append(row["customer_id"])
    return samples


@app.route("/")
def index():
    customer_id = request.args.get("customer", "").strip().upper()
    ollama_up = is_ollama_available()
    use_llm = request.args.get("llm", "1") == "1" and ollama_up
    result = history = summary = None
    if customer_id:
        with lock:
            result = rec_agent.recommend(customer_id, use_llm=use_llm)
            if "error" not in result:
                history = feedback_agent.get_recommendation_history(customer_id, limit=10)
                summary = feedback_agent.get_feedback_summary(customer_id)
    return render_template(
        "index.html", customer_id=customer_id, result=result, history=history, summary=summary,
        samples=sample_customers(), actions=ACTIONS, ollama_up=ollama_up,
        model=config.LLM_MODEL,
    )


@app.route("/feedback", methods=["POST"])
def feedback():
    customer_id = request.form["customer_id"]
    with lock:
        feedback_agent.log_feedback(customer_id, request.form["product_id"], request.form["action"])
    return redirect(url_for("index", customer=customer_id))


@app.route("/api/recommend/<customer_id>")
def api_recommend(customer_id):
    top_k = request.args.get("top", config.DEFAULT_TOP_K, type=int)
    with lock:
        result = rec_agent.recommend(customer_id.upper(), top_k=top_k, use_llm=is_ollama_available())
    if "error" in result:
        return jsonify(result), 404
    result["feedback_signals"] = {k: sorted(v) for k, v in result["feedback_signals"].items()}
    return jsonify(result)


if __name__ == "__main__":
    app.run(port=int(os.environ.get("PORT", 5000)), debug=False, threaded=False)
