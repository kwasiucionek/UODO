"""
Wywołania LLM — streaming, JSON, dekompozycja zapytania, lista modeli.

Obsługiwani providerzy:
  - Ollama — lokalny daemon (domyślnie http://localhost:11434).
             Modele cloud (np. "gpt-oss:120b-cloud") wymagają OLLAMA_CLOUD_API_KEY
             przekazywanego jako nagłówek Authorization — daemon używa go do
             uwierzytelnienia przy pobieraniu i uruchamianiu modeli z chmury.
             Modele lokalne (np. "gemma3") działają bez klucza, ale nagłówek
             i tak jest wysyłany — Ollama go po prostu ignoruje.
  - Groq   — zewnętrzne API (api.groq.com), wymaga GROQ_API_KEY.

LLM analizuje pytanie PRZED wyszukiwaniem zamiast szukać surowej frazy.
"""

import json as _json
import logging
from collections.abc import Generator
from typing import Any

import requests as _req
import streamlit as st

from config import (
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_PROVIDER,
    GROQ_API_KEY,
    OLLAMA_CLOUD_API_KEY,
    OLLAMA_URL,
)
from models import QueryDecomposition, QueryType

# ─────────────────────────── LISTA MODELI ────────────────────────


@st.cache_data(ttl=300, show_spinner=False)
def get_available_models(provider: str, api_key: str | None = None) -> list[str]:
    """Pobiera listę aktywnych modeli z API providera."""
    if provider == "Groq":
        try:
            from groq import Groq

            client = Groq(api_key=api_key or GROQ_API_KEY)
            ids = sorted(
                m.id
                for m in client.models.list().data
                if not any(x in m.id for x in ("whisper", "tts", "playai", "distil"))
            )
            return ids or ["llama-3.3-70b-versatile"]
        except Exception:
            return ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]

    # Ollama — zawsze localhost, klucz cloud w nagłówku
    try:
        r = _req.get(
            f"{OLLAMA_URL}/api/tags",
            headers=_ollama_headers(),
            timeout=5,
        )
        r.raise_for_status()
        models = [m.get("name") for m in r.json().get("models", []) if m.get("name")]
        return sorted(models) or [DEFAULT_OLLAMA_MODEL]
    except Exception:
        return [DEFAULT_OLLAMA_MODEL]


# ─────────────────────────── HELPERS ─────────────────────────────


def _ollama_headers() -> dict[str, str]:
    """Nagłówki dla każdego wywołania Ollama — klucz cloud jest zawsze wysyłany.
    Dla modeli czysto lokalnych Ollama ignoruje nagłówek Authorization.
    """
    return (
        {"Authorization": f"Bearer {OLLAMA_CLOUD_API_KEY}"}
        if OLLAMA_CLOUD_API_KEY
        else {}
    )


def _get_llm_params(
    provider: str | None, model: str | None, api_key: str | None
) -> tuple[str, str, str]:
    """Zwraca (provider, model, api_key) — z session_state jeśli nie podano."""
    return (
        provider or st.session_state.get("llm_provider", DEFAULT_PROVIDER),
        model or st.session_state.get("llm_model", ""),
        api_key or st.session_state.get("llm_api_key", ""),
    )


# ─────────────────────────── WYWOŁANIA LLM ───────────────────────


def call_llm_stream(
    query: str,
    context: str,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> Generator[str, Any, None]:
    """Stream odpowiedzi z Ollama (localhost) lub Groq."""
    provider, model, api_key = _get_llm_params(provider, model, api_key)

    system = (
        "Jesteś ekspertem ds. ochrony danych osobowych i prawa RODO. "
        "Pomagasz analizować decyzje Prezesa UODO oraz przepisy ustawy o ochronie danych osobowych. "
        "Odpowiadaj po polsku, precyzyjnie i zwięźle. "
        "Zawsze powołuj się na konkretne decyzje UODO podając sygnatury [np. DKN.XXXX.XX.XXXX, ZSOŚS, i in.] "
        "lub artykuły ustawy [np. Art. X u.o.d.o.]. "
        "Jeśli kontekst nie zawiera odpowiedzi na pytanie, powiedz o tym wprost."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Pytanie: {query}\n\nDokumenty:\n{context}"},
    ]

    if provider == "Groq":
        from groq import Groq

        client = Groq(api_key=api_key or GROQ_API_KEY)
        for chunk in client.chat.completions.create(  # type: ignore[call-overload]
            model=model or "",
            messages=messages,  # type: ignore[arg-type]
            max_tokens=2048,
            stream=True,
        ):
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
        return

    # Ollama — standardowe wywołanie localhost z kluczem cloud w nagłówku
    resp = _req.post(
        f"{OLLAMA_URL}/api/chat",
        headers=_ollama_headers(),
        json={"model": model, "messages": messages, "stream": True},
        stream=True,
        timeout=120,
    )
    for line in resp.iter_lines():
        if line:
            try:
                data = _json.loads(line)
                token = data.get("message", {}).get("content", "")
                if token:
                    yield token
                if data.get("done"):
                    break
            except Exception:
                pass


def call_llm_json(prompt, provider=None, model=None, api_key=None):
    import sys
    print(f"[call_llm_json] called, provider={provider}, model={model}", file=sys.stderr, flush=True)
    prompt: str,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Wywołanie LLM z wymaganym wyjściem JSON (bez streamowania)."""
    import logging
    import re as _re

    provider, model, api_key = _get_llm_params(provider, model, api_key)
    messages = [
        {
            "role": "system",
            "content": "Odpowiadaj WYŁĄCZNIE poprawnym JSON. Bez komentarzy. Bez markdown. Bez bloków ```json```.",
        },
        {"role": "user", "content": prompt},
    ]
    try:
        if provider == "Groq":
            from groq import Groq

            client = Groq(api_key=api_key or GROQ_API_KEY)
            resp = client.chat.completions.create(  # type: ignore[call-overload]
                model=model or "",
                messages=messages,  # type: ignore[arg-type]
                max_tokens=512,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content or "{}"
        else:
            r = _req.post(
                f"{OLLAMA_URL}/api/chat",
                headers=_ollama_headers(),
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "format": "json",
                },
                timeout=30,
            )
            raw = r.json().get("message", {}).get("content", "{}") or "{}"

        logging.warning(f"LLM JSON raw: {raw[:300]}")

        # Wyczyść markdown jeśli model go dodał
        raw = raw.strip()
        # Wyciągnij pierwszy blok JSON przez regex — najbardziej niezawodna metoda
        match = _re.search(r"\{.*\}", raw, _re.DOTALL)
        if match:
            raw = match.group(0)

        return _json.loads(raw)

    except Exception as e:
        logging.warning(f"call_llm_json failed: {e}")
    return {}


# ─────────────────────────── REASONING STEP ──────────────────────


def decompose_query(
    query: str,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> QueryDecomposition:
    """LLM analizuje pytanie i generuje ustrukturyzowane parametry wyszukiwania.
    Dla krótkich zapytań (≤ 3 słowa) zwraca uproszczoną dekompozycję bez wywołania LLM.
    """
    if len(query.strip().split()) <= 3:
        return QueryDecomposition(
            original_query=query,
            enriched_query=query,
            reasoning="Krótkie zapytanie — bez dekompozycji.",
        )

    prompt = f"""Jesteś ekspertem prawa ochrony danych osobowych.
Zanalizuj poniższe pytanie użytkownika i wygeneruj parametry wyszukiwania.

PYTANIE: "{query}"

Zwróć JSON w dokładnie takim formacie (wszystkie pola są wymagane):
{{
  "query_type": "szukam_decyzji" | "szukam_przepisu" | "analiza_ogólna" | "pytanie_faktyczne",
  "search_keywords": ["słowo1", "słowo2"],
  "gdpr_articles_hint": ["Art. 5", "Art. 83"],
  "uodo_act_articles_hint": ["Art. 60", "Art. 102"],
  "year_from_hint": null,
  "year_to_hint": null,
  "enriched_query": "rozszerzone zapytanie z synonimami prawnymi",
  "reasoning": "krótkie uzasadnienie po polsku (1 zdanie)"
}}

ZASADY:
- search_keywords: max 5, prawne synonimy (np. "kara" → ["administracyjna kara pieniężna", "sankcja"])
- enriched_query: rozszerz pytanie o kontekst prawny, nie zmieniaj sensu
- year_from_hint/year_to_hint: podaj rok tylko jeśli pytanie wyraźnie sugeruje okres
- artykuły: podaj tylko jeśli jesteś pewien, że są istotne dla pytania"""

    raw = call_llm_json(prompt, provider=provider, model=model, api_key=api_key)
    if not raw or "enriched_query" not in raw:
        return QueryDecomposition(
            original_query=query,
            enriched_query=query,
            reasoning="Dekompozycja niedostępna — używam oryginalnego zapytania.",
        )
    try:
        return QueryDecomposition(
            original_query=query,
            query_type=QueryType(raw.get("query_type", "analiza_ogólna")),
            search_keywords=raw.get("search_keywords", [])[:5],
            gdpr_articles_hint=raw.get("gdpr_articles_hint", []),
            uodo_act_articles_hint=raw.get("uodo_act_articles_hint", []),
            year_from_hint=raw.get("year_from_hint"),
            year_to_hint=raw.get("year_to_hint"),
            enriched_query=raw.get("enriched_query", query),
            reasoning=raw.get("reasoning", ""),
        )
    except Exception:
        return QueryDecomposition(
            original_query=query,
            enriched_query=query,
            reasoning="Błąd parsowania — używam oryginalnego zapytania.",
        )
