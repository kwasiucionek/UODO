#!/usr/bin/env python3
"""
enrich_act_keywords.py — generuje keywords dla artykułów u.o.d.o. i RODO przez LLM
i aktualizuje payloady bezpośrednio w Qdrant (bez przeindeksowania).

Uruchomienie:
  python enrich_act_keywords.py --provider ollama --model qwen3:14b
  python enrich_act_keywords.py --provider groq --model llama-3.3-70b-versatile
  python enrich_act_keywords.py --dry-run   # tylko wypisz, nie zapisuj

Opcje:
  --qdrant      URL Qdrant (domyślnie http://localhost:6333)
  --collection  Nazwa kolekcji (domyślnie uodo_decisions)
  --provider    ollama lub groq
  --model       Nazwa modelu
  --api-key     Klucz API (lub z .env)
  --doc-types   Typy do wzbogacenia (domyślnie: legal_act_article gdpr_article gdpr_recital)
  --dry-run     Tylko wypisz, nie zapisuj do Qdrant
  --delay       Opóźnienie między zapytaniami w sekundach (domyślnie 0.5)
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OLLAMA_CLOUD_API_KEY = os.getenv("OLLAMA_CLOUD_API_KEY", "")
OLLAMA_CLOUD_URL = os.getenv("OLLAMA_CLOUD_URL", "https://ollama.com")
OLLAMA_LOCAL_URL = os.getenv("OLLAMA_LOCAL_URL", "http://localhost:11434")

# Istniejące tagi z decyzji UODO — LLM ma wybierać z tej listy jeśli pasują
# (pobierane dynamicznie z Qdrant)


def get_existing_tags(client, collection: str) -> List[str]:
    """Pobiera wszystkie unikalne tagi z decyzji UODO."""
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    all_tags = set()
    offset = None
    while True:
        pts, next_off = client.scroll(
            collection_name=collection,
            limit=500,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="doc_type", match=MatchValue(value="uodo_decision")
                    )
                ]
            ),
            with_payload=["keywords"],
            with_vectors=False,
            offset=offset,
        )
        for pt in pts or []:
            kws = (pt.payload or {}).get("keywords", [])
            if isinstance(kws, list):
                all_tags.update(kws)
        if not next_off or not pts:
            break
        offset = next_off
    return sorted(all_tags)


def call_llm(prompt: str, provider: str, model: str, api_key: str) -> str:
    if provider == "groq":
        from groq import Groq

        client = Groq(api_key=api_key or GROQ_API_KEY)
        resp = client.chat.completions.create(
            model=model,
            max_tokens=200,
            stream=False,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or ""

    elif provider == "ollama":
        import requests

        # Spróbuj lokalnie, potem cloud
        for base_url in [OLLAMA_LOCAL_URL, OLLAMA_CLOUD_URL]:
            try:
                headers = {}
                if base_url == OLLAMA_CLOUD_URL:
                    headers["Authorization"] = (
                        f"Bearer {api_key or OLLAMA_CLOUD_API_KEY}"
                    )
                resp = requests.post(
                    f"{base_url}/api/chat",
                    headers=headers,
                    json={
                        "model": model,
                        "stream": False,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                    timeout=60,
                )
                return resp.json().get("message", {}).get("content", "")
            except Exception:
                continue
    return ""


def generate_keywords(
    article_num: str,
    content: str,
    doc_type: str,
    existing_tags: List[str],
    provider: str,
    model: str,
    api_key: str,
) -> List[str]:
    """Generuje keywords dla artykułu przez LLM."""

    if doc_type == "legal_act_article":
        source = "ustawy o ochronie danych osobowych (u.o.d.o.)"
        label = f"Art. {article_num} u.o.d.o."
    elif doc_type == "gdpr_article":
        source = "RODO (rozporządzenie UE 2016/679)"
        label = f"Art. {article_num} RODO"
    else:
        source = "RODO — motywy preambuły"
        label = f"Motyw {article_num} RODO"

    # Skróć treść do pierwszych 800 znaków
    content_short = content[:800].strip()

    tags_sample = "\n".join(f"- {t}" for t in existing_tags[:80])

    prompt = (
        f"Jesteś ekspertem prawa ochrony danych osobowych.\n"
        f"Poniżej jest treść {label} {source}.\n\n"
        f"Wygeneruj 6-10 słów kluczowych (tagów) opisujących ten przepis.\n"
        f"Wybieraj tylko tagi z poniższej listy (użyj dokładnie tej samej pisowni).\n"
        f"Możesz dodać nowe tagi tylko jeśli żaden z listy nie pasuje.\n"
        f"Odpowiedz TYLKO listą tagów, jeden na linię, bez komentarzy.\n\n"
        f"Treść przepisu:\n{content_short}\n\n"
        f"Dostępne tagi (wybierz pasujące):\n{tags_sample}"
    )

    raw = call_llm(prompt, provider, model, api_key)
    keywords = []
    for line in raw.strip().splitlines():
        tag = line.strip().lstrip("- •*").strip()
        if tag and len(tag) > 2 and len(tag) < 80:
            keywords.append(tag)
    return keywords[:6]


def enrich_documents(
    qdrant_url: str,
    collection: str,
    provider: str,
    model: str,
    api_key: str,
    doc_types: List[str],
    dry_run: bool,
    delay: float,
):
    from qdrant_client import QdrantClient
    from qdrant_client.models import FieldCondition, Filter, MatchAny

    client = QdrantClient(url=qdrant_url, timeout=60)

    print("Pobieranie istniejących tagów z decyzji UODO...")
    existing_tags = get_existing_tags(client, collection)
    print(f"Znaleziono {len(existing_tags)} unikalnych tagów")

    # Pobierz wszystkie artykuły bez keywords
    print(f"\nPobieranie dokumentów typów: {doc_types}...")
    docs_to_enrich = []
    offset = None
    while True:
        pts, next_off = client.scroll(
            collection_name=collection,
            scroll_filter=Filter(
                must=[FieldCondition(key="doc_type", match=MatchAny(any=doc_types))]
            ),
            limit=200,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for pt in pts or []:
            pay = pt.payload or {}
            kws = pay.get("keywords", [])
            # Pomiń już wzbogacone
            if kws:
                continue
            docs_to_enrich.append((str(pt.id), pay))
        if not next_off or not pts:
            break
        offset = next_off

    total = len(docs_to_enrich)
    print(f"Dokumentów do wzbogacenia: {total}")
    if total == 0:
        print("Wszystkie dokumenty mają już keywords.")
        return

    ok, errors = 0, 0
    for i, (point_id, payload) in enumerate(docs_to_enrich, 1):
        art_num = payload.get("article_num", "?")
        dtype = payload.get("doc_type", "")
        content = payload.get("content_text", "")

        print(f"[{i}/{total}] {dtype} — {art_num} ... ", end="", flush=True)

        try:
            keywords = generate_keywords(
                str(art_num), content, dtype, existing_tags, provider, model, api_key
            )
        except Exception as e:
            print(f"BŁĄD LLM: {e}")
            errors += 1
            continue

        if not keywords:
            print("brak tagów")
            continue

        print(f"{keywords}")

        if not dry_run:
            try:
                client.set_payload(
                    collection_name=collection,
                    payload={
                        "keywords": keywords,
                        "keywords_text": ", ".join(keywords),
                    },
                    points=[point_id],
                )
                ok += 1
            except Exception as e:
                print(f"  BŁĄD zapisu: {e}")
                errors += 1
        else:
            ok += 1

        if delay > 0:
            time.sleep(delay)

    print(f"\nGotowe! Wzbogacono: {ok}, błędy: {errors}")
    if dry_run:
        print("(dry-run — nic nie zapisano)")


def main():
    parser = argparse.ArgumentParser(
        description="Generuje keywords dla artykułów u.o.d.o. i RODO"
    )
    parser.add_argument("--qdrant", default=QDRANT_URL)
    parser.add_argument("--collection", default="uodo_decisions")
    parser.add_argument("--provider", default="ollama", choices=["ollama", "groq"])
    parser.add_argument("--model", default="qwen3:14b")
    parser.add_argument("--api-key", default="")
    parser.add_argument(
        "--doc-types",
        nargs="+",
        default=["legal_act_article", "gdpr_article", "gdpr_recital"],
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--delay", type=float, default=0.5)
    args = parser.parse_args()

    enrich_documents(
        qdrant_url=args.qdrant,
        collection=args.collection,
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        doc_types=args.doc_types,
        dry_run=args.dry_run,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()
