from __future__ import annotations
import logging
import time
from functools import lru_cache
from scholaragent import config as cfg
from scholaragent.utils import parse_json

log = logging.getLogger(__name__)

DEFAULT_MODELS = {"gemini": "gemini-2.5-flash", "groq": "llama-3.3-70b-versatile", "ollama": "llama3.1"}

@lru_cache(maxsize=4)
def get_chat_model(temperature: float = 0.0):
    provider = cfg.LLM_PROVIDER.lower()
    if provider not in DEFAULT_MODELS:
        raise ValueError(f"Unknown LLM_PROVIDER={provider!r}; use one of {sorted(DEFAULT_MODELS)}")
    model = cfg.LLM_MODEL or DEFAULT_MODELS[provider]
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, temperature=temperature)
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=model, temperature=temperature)
    from langchain_ollama import ChatOllama
    return ChatOllama(model=model, temperature=temperature)

def _to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict) and p.get("type") == "text":
                parts.append(p.get("text", ""))
        return "".join(parts)
    return str(content)

def ask_llm(prompt: str, system: str | None = None, temperature: float = 0.0, retries: int = 3) -> str:
    messages = ([("system", system)] if system else []) + [("human", prompt)]
    for attempt in range(retries):
        try:
            return _to_text(get_chat_model(temperature).invoke(messages).content).strip()
        except Exception as exc:
            if attempt == retries - 1:
                raise
            wait = 2 * 2**attempt
            log.warning("LLM call failed (%s). Retrying in %ss", exc, wait)
            time.sleep(wait)
    raise RuntimeError("unreachable")

def ask_json(prompt: str, system: str | None = None) -> dict:
    raw = ask_llm(prompt + "\n\nReturn ONLY valid JSON.", system)
    try:
        return parse_json(raw)
    except ValueError:
        raw = ask_llm(
            prompt + "\n\nYour previous reply was not valid JSON. Return ONLY a JSON object: no prose, no code fences.",
            system,
        )
        return parse_json(raw)