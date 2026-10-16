# utils/ollama_interface.py
#
# Thin HTTP client for a local Ollama server. Agents receive `ollama_llm` as `llm_func`,
# so swapping the model or backend only touches this file.

import requests

import config


class LLMUnavailable(RuntimeError):
    """Ollama is not reachable or returned an error. Agents fall back to rule-based output."""


def is_ollama_available(timeout=2):
    try:
        return requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=timeout).ok
    except requests.RequestException:
        return False


def ollama_llm(prompt, model=None, json_mode=False):
    payload = {
        "model": model or config.LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2},
    }
    if json_mode:
        payload["format"] = "json"
    try:
        response = requests.post(
            f"{config.OLLAMA_URL}/api/generate", json=payload, timeout=config.LLM_TIMEOUT
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except (requests.RequestException, ValueError) as e:
        raise LLMUnavailable(f"Ollama generate failed: {e}") from e


def ollama_embed(texts, model=None):
    """Return one embedding (list of floats) per input text."""
    try:
        response = requests.post(
            f"{config.OLLAMA_URL}/api/embed",
            json={"model": model or config.EMBED_MODEL, "input": list(texts)},
            timeout=config.LLM_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()["embeddings"]
    except (requests.RequestException, ValueError, KeyError) as e:
        raise LLMUnavailable(f"Ollama embed failed: {e}") from e
