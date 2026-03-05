#!/usr/bin/env python3
"""
UODO Indexer — embedduje i indeksuje orzeczenia UODO w Qdrant.

Kolekcja: uodo_decisions
Użycie:
  python uodo_indexer.py --jsonl uodo_decisions.jsonl
  python uodo_indexer.py --jsonl uodo_decisions.jsonl --rebuild
"""

import argparse
import hashlib
import json
import uuid
from typing import Any, Dict, List

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, FieldCondition, Filter, MatchValue,
    PayloadSchemaType, PointStruct, VectorParams,
)
from sentence_transformers import SentenceTransformer

# ─────────────────────────── CONFIG ──────────────────────────────

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "uodo_decisions"
EMBED_MODEL = "sdadas/mmlw-retrieval-roberta-large"
EMBED_MAX_CHARS = 5500
BATCH_SIZE = 50


# ─────────────────────────── HELPERS ─────────────────────────────

def load_embedder(device: str = None) -> SentenceTransformer:
    import torch
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🤖 Ładuję embedder: {EMBED_MODEL} ({device})")
    return SentenceTransformer(EMBED_MODEL, device=device, trust_remote_code=True)


def sig_to_uuid(sig: str) -> str:
    return str(uuid.UUID(bytes=hashlib.md5(f"uodo:{sig}".encode()).digest()))


def build_embed_text(doc: Dict) -> str:
    """
    Tekst do embeddingu — kolejność ważna:
    1. Sygnatura + tytuł pełny (opis naruszenia) — najwyższa waga
    2. Keywords — tagi UODO z baz terminologicznych
    3. Podmiot decyzji (entities)
    4. Fragment treści
    5. Powołane akty
    """
    sig = doc.get("signature", "")
    title_full = doc.get("title_full", "") or doc.get("title", "")
    status = doc.get("status", "")
    keywords = doc.get("keywords", "")
    content = doc.get("content_text", "")[:EMBED_MAX_CHARS]

    # Podmiot ukarany z entities (function=other lub author)
    subject = " ".join(
        e.get("name", "") or e.get("title", "")
        for e in doc.get("entities", [])
        if e.get("function") in ("other", "author")
    )

    refs = doc.get("refs_from_content", {})
    acts = " ".join(refs.get("acts", [])[:5])
    eu = " ".join(refs.get("eu_acts", [])[:3])

    return (
        f"{sig} {title_full} {status}\n"
        f"Słowa kluczowe: {keywords}\n"
        f"Podmiot: {subject}\n\n"
        f"{content}\n\n"
        f"Akty: {acts} {eu}"
    )


def build_payload(doc: Dict) -> Dict[str, Any]:
    refs = doc.get("refs_from_content", {})

    # Deduplikacja: łącz refs z treści i z related_legislation/rulings
    acts = list(dict.fromkeys(
        refs.get("acts", []) +
        [r["signature"] for r in doc.get("related_legislation", [])
         if r.get("type") == "act"]
    ))
    eu_acts = list(dict.fromkeys(
        refs.get("eu_acts", []) +
        [r["signature"] for r in doc.get("related_legislation", [])
         if r.get("type") == "eu_act"]
    ))
    uodo_rulings = list(dict.fromkeys(
        refs.get("uodo_rulings", []) +
        [r["signature"] for r in doc.get("related_rulings", [])
         if r.get("type") == "uodo_ruling"]
    ))
    court_rulings = list(dict.fromkeys(
        refs.get("court_rulings", []) +
        [r["signature"] for r in doc.get("related_rulings", [])
         if r.get("type") == "court_ruling"]
    ))

    # Keywords jako lista (do filtrowania w Qdrant) i string (do wyświetlania)
    kw_list = doc.get("keywords_list", [])
    if not kw_list:
        kw_raw = doc.get("keywords", "")
        kw_list = [k.strip() for k in kw_raw.split(",") if k.strip()]
    kw_text = doc.get("keywords", "") or ", ".join(kw_list)

    # Podmiot ukarany z entities
    subject = " | ".join(
        e.get("name", "") or e.get("title", "")
        for e in doc.get("entities", [])
        if e.get("function") in ("other", "author") and (e.get("name") or e.get("title"))
    )

    # Typy relacji do aktów (dla grafu — np. implements RODO, quotes ustawa)
    refs_full = doc.get("refs_full", [])
    relation_map = {r.get("signature", ""): r.get("relation", "refers")
                    for r in refs_full}

    return {
        "doc_type": "uodo_decision",
        "signature": doc.get("signature", ""),
        "refid": doc.get("refid", ""),
        "title": doc.get("title", ""),
        "title_full": doc.get("title_full", "") or doc.get("title", ""),
        "kind": doc.get("kind", "decision"),
        "status": doc.get("status", ""),
        "pub_workflow_status": doc.get("pub_workflow_status", ""),
        "date_issued": doc.get("date_issued", ""),
        "date_published": doc.get("date_published", ""),
        "date_effect": doc.get("date_effect", ""),
        "year": doc.get("year", 0),
        "content_text": doc.get("content_text", "")[:50000],
        "source_url": doc.get("url", ""),
        "source_collection": "UODO",
        # Keywords — lista do filtrowania po tagach w Qdrant
        "keywords": kw_list,
        "keywords_text": kw_text,
        # Podmiot decyzji
        "subject": subject,
        # Powiązania — krawędzie grafu
        "related_acts": acts[:50],
        "related_eu_acts": eu_acts[:20],
        "related_uodo_rulings": uodo_rulings[:30],
        "related_court_rulings": court_rulings[:20],
        # Typy relacji (quotes / refers / implements / amends)
        "relation_map": json.dumps(
            {sig: rel for sig, rel in relation_map.items() if sig},
            ensure_ascii=False
        )[:2000],
        # Taksonomia UODO — rozbita wg grup labelów
        "term_decision_type":      doc.get("term_decision_type", []),
        "term_violation_type":     doc.get("term_violation_type", []),
        "term_legal_basis":        doc.get("term_legal_basis", []),
        "term_corrective_measure": doc.get("term_corrective_measure", []),
        "term_sector":             doc.get("term_sector", []),
    }


# ─────────────────────────── MAIN ────────────────────────────────

def index_decisions(jsonl_path: str, qdrant_url: str = QDRANT_URL,
                    rebuild: bool = False, device: str = None):

    docs = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    docs.append(json.loads(line))
                except Exception:
                    pass
    print(f"📋 {len(docs)} dokumentów")

    model = load_embedder(device)
    dim = model.get_sentence_embedding_dimension()
    print(f"📐 Wymiar wektora: {dim}")

    client = QdrantClient(url=qdrant_url, timeout=60)
    existing = {c.name for c in client.get_collections().collections}

    if COLLECTION_NAME in existing and rebuild:
        print(f"🗑️  Usuwam kolekcję: {COLLECTION_NAME}")
        client.delete_collection(COLLECTION_NAME)

    if COLLECTION_NAME not in existing or rebuild:
        print(f"📦 Tworzę kolekcję: {COLLECTION_NAME} (dim={dim})")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        for field, schema in [
            ("signature",           PayloadSchemaType.KEYWORD),
            ("status",              PayloadSchemaType.KEYWORD),
            ("kind",                PayloadSchemaType.KEYWORD),
            ("pub_workflow_status", PayloadSchemaType.KEYWORD),
            ("year",                PayloadSchemaType.INTEGER),
            ("keywords",            PayloadSchemaType.KEYWORD),  # filtrowanie po tagach
            ("doc_type",            PayloadSchemaType.KEYWORD),
            ("term_decision_type",      PayloadSchemaType.KEYWORD),
            ("term_violation_type",     PayloadSchemaType.KEYWORD),
            ("term_legal_basis",        PayloadSchemaType.KEYWORD),
            ("term_corrective_measure", PayloadSchemaType.KEYWORD),
            ("term_sector",             PayloadSchemaType.KEYWORD),
        ]:
            client.create_payload_index(COLLECTION_NAME, field, schema)
    else:
        print(f"✅ Kolekcja istnieje")

    # Sprawdź już zaindeksowane
    done = set()
    if not rebuild:
        offset = None
        while True:
            pts, next_off = client.scroll(
                collection_name=COLLECTION_NAME, limit=500, offset=offset,
                with_payload=["signature"], with_vectors=False,
            )
            for p in pts:
                done.add((p.payload or {}).get("signature", ""))
            if next_off is None:
                break
            offset = next_off
        print(f"🔄 Już zaindeksowane: {len(done)}")

    to_index = [d for d in docs if d.get("signature") not in done]
    print(f"📝 Do zaindeksowania: {len(to_index)}")
    if not to_index:
        print("✅ Gotowe!")
        return

    errors = indexed = 0
    for batch_start in range(0, len(to_index), BATCH_SIZE):
        batch = to_index[batch_start:batch_start + BATCH_SIZE]
        points = []
        for doc in batch:
            sig = doc.get("signature", "")
            if not sig:
                continue
            try:
                vec = model.encode(
                    build_embed_text(doc), normalize_embeddings=True
                ).tolist()
            except Exception as e:
                print(f"  ⚠️ Embedding błąd {sig}: {e}")
                errors += 1
                continue
            points.append(PointStruct(
                id=sig_to_uuid(sig),
                vector=vec,
                payload=build_payload(doc),
            ))
        if not points:
            continue
        try:
            client.upsert(collection_name=COLLECTION_NAME, points=points)
            indexed += len(points)
            pct = (batch_start + len(batch)) / len(to_index) * 100
            print(f"  ✅ +{len(points)} ({pct:.0f}%)")
        except Exception as e:
            print(f"  ❌ Upsert błąd: {e}")
            errors += len(points)

    print(f"\n✅ Zaindeksowano: {indexed}, błędy: {errors}")
    info = client.get_collection(COLLECTION_NAME)
    print(f"📊 Kolekcja: {info.points_count} punktów")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", default="uodo_decisions.jsonl")
    parser.add_argument("--qdrant", default=QDRANT_URL)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    index_decisions(args.jsonl, args.qdrant, args.rebuild, args.device)
