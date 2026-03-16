#!/usr/bin/env python3
"""
UODO Act Indexer — indeksuje ustawę o ochronie danych osobowych (Dz.U. 2019 poz. 1781)
do tej samej kolekcji Qdrant co orzeczenia UODO.

Każdy artykuł = osobny dokument (doc_type="legal_act_article").
Długie artykuły są dzielone na chunki z zachłannym overlapem.

Użycie:
  python uodo_act_indexer.py --md D20191781L.md
  python uodo_act_indexer.py --md D20191781L.md --rebuild
"""

import argparse
import hashlib
import re
import uuid
from typing import Dict, List, Tuple

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, PayloadSchemaType, PointStruct, VectorParams,
)
from sentence_transformers import SentenceTransformer

# ─────────────────────────── CONFIG ──────────────────────────────

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "uodo_decisions"   # ta sama kolekcja co orzeczenia
EMBED_MODEL = "sdadas/mmlw-retrieval-roberta-large"

SIGNATURE = "Dz.U. 2019 poz. 1781"
ACT_TITLE = "Ustawa z dnia 10 maja 2018 r. o ochronie danych osobowych"
ACT_SHORT = "u.o.d.o."

# Artykuły 1–108 to właściwa ustawa; 109+ to przepisy zmieniające inne ustawy
ART_MAX = 108

# Chunking — artykuły dłuższe niż MAX_CHUNK_CHARS są dzielone
MAX_CHUNK_CHARS = 3000
CHUNK_OVERLAP_CHARS = 300

BATCH_SIZE = 50


# ─────────────────────────── PARSOWANIE MD ───────────────────────

def parse_articles(md_path: str) -> List[Dict]:
    """
    Parsuje plik markdown i wyciąga artykuły 1–ART_MAX właściwej ustawy.
    Pomija wstępne przepisy zmieniające (Art. 109–157 na początku pliku).
    Zwraca listę: {article_num, article_text, header_line}
    """
    with open(md_path, encoding="utf-8") as f:
        lines = f.readlines()

    # Znajdź linię gdzie zaczyna się właściwa ustawa (Art. 1. 1. Ustawę stosuje się...)
    start_line = 0
    for i, line in enumerate(lines):
        if re.match(r'^Art\. 1\. 1\. Ustawę stosuje się', line):
            start_line = i
            break

    if start_line == 0:
        raise ValueError("Nie znaleziono Art. 1 właściwej ustawy w pliku MD")

    # Wyciągnij tylko właściwą ustawę
    content_lines = lines[start_line:]
    content = "".join(content_lines)

    # Podziel po nagłówkach artykułów: "Art. X."  (X = liczba)
    article_pattern = re.compile(r'(?=^Art\. (\d+)\.)', re.MULTILINE)
    parts = article_pattern.split(content)

    articles = []
    i = 0
    while i < len(parts):
        part = parts[i]
        # Sprawdź czy to numer artykułu
        if re.match(r'^\d+$', part.strip()):
            art_num = int(part.strip())
            art_text = parts[i + 1] if i + 1 < len(parts) else ""
            i += 2

            if art_num > ART_MAX:
                continue  # pomiń przepisy zmieniające

            # Oczyść tekst — usuń nagłówki stron "## Strona X"
            art_text = re.sub(r'---\s*\n## Strona \d+\s*\n', '\n', art_text)
            art_text = art_text.strip()

            # Wyciągnij pierwsze zdanie jako "tytuł" (do embeddingu)
            first_line = art_text.split('\n')[0][:200]

            articles.append({
                "article_num": art_num,
                "article_text": f"Art. {art_num}. {art_text}",
                "first_line": first_line,
            })
        else:
            i += 1

    print(f"📜 Sparsowano {len(articles)} artykułów (Art. 1–{ART_MAX})")
    return articles


def chunk_article(art: Dict) -> List[Dict]:
    """
    Dzieli długi artykuł na chunki. Krótkie artykuły zwracane jako jeden chunk.
    Każdy chunk dostaje indeks i informację o artykule.
    """
    text = art["article_text"]
    if len(text) <= MAX_CHUNK_CHARS:
        return [{**art, "chunk_index": 0, "chunk_total": 1, "chunk_text": text}]

    # Dziel po liniach paragrafów (§ X lub pkt X lub tiret)
    # Staraj się nie rozrywać paragrafów
    chunks = []
    current = ""
    chunk_idx = 0

    paragraphs = re.split(r'(\n(?=\d+\)|§|\-\s|[a-z]\)))', text)

    for para in paragraphs:
        if len(current) + len(para) <= MAX_CHUNK_CHARS:
            current += para
        else:
            if current.strip():
                chunks.append(current.strip())
            # Overlap — weź ostatnie CHUNK_OVERLAP_CHARS z poprzedniego chunka
            overlap = current[-CHUNK_OVERLAP_CHARS:] if len(current) > CHUNK_OVERLAP_CHARS else current
            current = overlap + para

    if current.strip():
        chunks.append(current.strip())

    result = []
    for idx, chunk_text in enumerate(chunks):
        result.append({
            **art,
            "chunk_index": idx,
            "chunk_total": len(chunks),
            "chunk_text": chunk_text,
        })
    return result


# ─────────────────────────── EMBEDDING ───────────────────────────

def build_embed_text(chunk: Dict) -> str:
    """
    Tekst do embeddingu: kontekst aktu + numer artykułu + treść.
    Kontekst na początku daje wyższą wagę przy wyszukiwaniu.
    """
    art_num = chunk["article_num"]
    chunk_idx = chunk["chunk_index"]
    total = chunk["chunk_total"]
    text = chunk["chunk_text"]

    header = (
        f"{SIGNATURE} {ACT_SHORT} {ACT_TITLE}\n"
        f"Art. {art_num}"
    )
    if total > 1:
        header += f" (część {chunk_idx + 1}/{total})"
    header += "\n\n"

    return header + text


def sig_to_uuid(key: str) -> str:
    return str(uuid.UUID(bytes=hashlib.md5(key.encode()).digest()))


# ─────────────────────────── MAIN ────────────────────────────────

def index_act(md_path: str, qdrant_url: str = QDRANT_URL,
              rebuild_act: bool = False, device: str = None):

    import torch
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"🤖 Ładuję embedder: {EMBED_MODEL} ({device})")
    model = SentenceTransformer(EMBED_MODEL, device=device, trust_remote_code=True)
    dim = model.get_sentence_embedding_dimension()

    client = QdrantClient(url=qdrant_url, timeout=60)
    existing = {c.name for c in client.get_collections().collections}

    # Utwórz kolekcję jeśli nie istnieje (ten sam schemat co uodo_indexer.py)
    if COLLECTION_NAME not in existing:
        print(f"📦 Tworzę kolekcję: {COLLECTION_NAME} (dim={dim})")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        for field, schema in [
            ("signature", PayloadSchemaType.KEYWORD),
            ("status", PayloadSchemaType.KEYWORD),
            ("year", PayloadSchemaType.INTEGER),
            ("keywords", PayloadSchemaType.KEYWORD),
            ("doc_type", PayloadSchemaType.KEYWORD),
        ]:
            client.create_payload_index(COLLECTION_NAME, field, schema)
    else:
        print(f"✅ Kolekcja {COLLECTION_NAME} istnieje")

    # Sprawdź które artykuły już są zaindeksowane
    done_keys = set()
    if not rebuild_act:
        offset = None
        while True:
            pts, next_off = client.scroll(
                collection_name=COLLECTION_NAME, limit=500, offset=offset,
                with_payload=["doc_id"], with_vectors=False,
            )
            for p in pts:
                doc_id = (p.payload or {}).get("doc_id", "")
                if doc_id.startswith("uodo_act:"):
                    done_keys.add(doc_id)
            if next_off is None:
                break
            offset = next_off
        if done_keys:
            print(f"🔄 Już zaindeksowane artykuły: {len(done_keys)}")

    # Parsuj artykuły i podziel na chunki
    articles = parse_articles(md_path)
    all_chunks = []
    for art in articles:
        all_chunks.extend(chunk_article(art))

    print(f"📝 Chunki do zaindeksowania: {len(all_chunks)} "
          f"(z {len(articles)} artykułów)")

    to_index = [
        c for c in all_chunks
        if f"uodo_act:{SIGNATURE}:art{c['article_num']}:chunk{c['chunk_index']}" not in done_keys
    ]
    print(f"📝 Nowych: {len(to_index)}")

    if not to_index:
        print("✅ Wszystkie artykuły już zaindeksowane!")
        return

    errors = indexed = 0
    for batch_start in range(0, len(to_index), BATCH_SIZE):
        batch = to_index[batch_start:batch_start + BATCH_SIZE]
        points = []

        for chunk in batch:
            art_num = chunk["article_num"]
            chunk_idx = chunk["chunk_index"]
            total = chunk["chunk_total"]
            doc_id = f"uodo_act:{SIGNATURE}:art{art_num}:chunk{chunk_idx}"

            try:
                vec = model.encode(
                    build_embed_text(chunk), normalize_embeddings=True
                ).tolist()
            except Exception as e:
                print(f"  ⚠️ Embedding błąd Art. {art_num}: {e}")
                errors += 1
                continue

            payload = {
                "doc_type": "legal_act_article",
                "doc_id": doc_id,
                "signature": SIGNATURE,
                "title": ACT_TITLE,
                "act_short": ACT_SHORT,
                "article_num": art_num,
                "article_label": f"Art. {art_num}",
                "chunk_index": chunk_idx,
                "chunk_total": total,
                "content_text": chunk["chunk_text"],
                "status": "obowiązujący",
                "year": 2019,
                "source_collection": "UODO_ACT",
                "source_url": f"https://isap.sejm.gov.pl/isap.nsf/DocDetails.xsp?id=WDU20190001781",
                # Pola kompatybilne z kartą orzeczenia (dla ujednoliconego wyświetlania)
                "keywords": [],
                "keywords_text": "",
                "related_acts": [],
                "related_eu_acts": ["EU 2016/679"],
                "related_uodo_rulings": [],
                "related_court_rulings": [],
            }

            points.append(PointStruct(
                id=sig_to_uuid(doc_id),
                vector=vec,
                payload=payload,
            ))

        if not points:
            continue

        try:
            client.upsert(collection_name=COLLECTION_NAME, points=points)
            indexed += len(points)
            pct = (batch_start + len(batch)) / len(to_index) * 100
            print(f"  ✅ +{len(points)} chunków ({pct:.0f}%)")
        except Exception as e:
            print(f"  ❌ Upsert błąd: {e}")
            errors += len(points)

    print(f"\n✅ Zaindeksowano: {indexed} chunków, błędy: {errors}")
    info = client.get_collection(COLLECTION_NAME)
    print(f"📊 Kolekcja łącznie: {info.points_count} punktów")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Indeksuje ustawę o ochronie danych osobowych do Qdrant"
    )
    parser.add_argument("--md", required=True, help="Ścieżka do pliku .md ustawy")
    parser.add_argument("--qdrant", default=QDRANT_URL)
    parser.add_argument("--rebuild-act", action="store_true",
                        help="Usuń i przeindeksuj artykuły (nie dotyka orzeczeń)")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    index_act(args.md, args.qdrant, args.rebuild_act, args.device)
