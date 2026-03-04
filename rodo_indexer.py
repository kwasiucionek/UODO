#!/usr/bin/env python3
"""
rodo_indexer.py — parsuje tekst RODO (2016/679) z pliku Markdown i indeksuje go w Qdrant.

Uruchomienie:
  python rodo_indexer.py
  python rodo_indexer.py --md rodo_2016_679_pl.md
  python rodo_indexer.py --qdrant http://localhost:6333 --collection uodo_decisions

Opcje:
  --md          Ścieżka do pliku Markdown (domyślnie rodo_2016_679_pl.md)
  --qdrant      URL Qdrant (domyślnie http://localhost:6333)
  --collection  Nazwa kolekcji (domyślnie uodo_decisions)
  --model       Model embeddingowy (domyślnie sdadas/mmlw-retrieval-roberta-large)
  --dry-run     Parsuj i wypisz bez indeksowania
"""

import argparse
import re
import sys
import time
import uuid
from pathlib import Path
from typing import List, Dict

# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_MD   = "rodo_2016_679_pl.md"
RODO_SOURCE  = "RODO 2016/679"
RODO_URL     = "https://eur-lex.europa.eu/legal-content/PL/TXT/?uri=CELEX:32016R0679"

CHUNK_MAX_CHARS = 1200
CHUNK_OVERLAP   = 100


# ─────────────────────────── CHUNKING ────────────────────────────────────────

def split_into_chunks(text: str, max_chars: int = CHUNK_MAX_CHARS,
                      overlap: int = CHUNK_OVERLAP) -> List[str]:
    if len(text) <= max_chars:
        return [text]
    paragraphs = re.split(r"\n\n+", text)
    chunks, current, current_len = [], [], 0
    for para in paragraphs:
        para_len = len(para)
        if current_len + para_len > max_chars and current:
            chunks.append("\n\n".join(current))
            overlap_paras, overlap_len = [], 0
            for p in reversed(current):
                if overlap_len + len(p) <= overlap:
                    overlap_paras.insert(0, p)
                    overlap_len += len(p)
                else:
                    break
            current, current_len = overlap_paras, overlap_len
        current.append(para)
        current_len += para_len
    if current:
        chunks.append("\n\n".join(current))
    return chunks


# ─────────────────────────── PARSOWANIE MD ───────────────────────────────────

def parse_rodo_md(text: str) -> List[Dict]:
    """
    Parsuje plik Markdown RODO.
    Motywy: linie zaczynające się od '- (N) tekst'
    Artykuły: nagłówki '#+ Artykuł N', tytuł to kolejny niepusty nagłówek
    """
    lines = text.splitlines()
    documents = []

    # ── Motywy ────────────────────────────────────────────────
    recital_re = re.compile(r"^- \((\d{1,3})\)\s+(.*)")
    recitals: Dict[str, str] = {}
    current_recital = None

    for line in lines:
        m = recital_re.match(line)
        if m:
            current_recital = m.group(1)
            recitals[current_recital] = m.group(2).strip()
        elif current_recital:
            if re.match(r"^#{1,6}\s+ROZDZIAŁ", line):
                current_recital = None
            elif line.startswith("> ") or re.match(r"^#{1,6}\s+", line):
                pass  # pomiń przypisy i nagłówki
            elif line.strip():
                recitals[current_recital] += " " + line.lstrip("- ").strip()

    for num, content in recitals.items():
        content = content.strip()
        if len(content) < 20:
            continue
        documents.append({
            "doc_type": "gdpr_recital",
            "article_num": f"motyw {num}",
            "chapter": "Preambuła",
            "chapter_title": "Motywy",
            "content_text": f"Motyw {num} RODO:\n{content}",
            "source": RODO_SOURCE,
            "source_url": RODO_URL,
            "chunk_index": 0,
            "chunk_total": 1,
        })

    # ── Rozdziały i artykuły ──────────────────────────────────
    chapter_id, chapter_title = "", ""
    i = 0
    art_count = 0

    while i < len(lines):
        line = lines[i]

        # Rozdział
        m_ch = re.match(r"^#{1,6}\s+ROZDZIAŁ\s+([IVXLC]+)\s*$", line)
        if m_ch:
            chapter_id = m_ch.group(1)
            chapter_title = ""
            for j in range(i + 1, min(i + 5, len(lines))):
                t = lines[j].strip()
                if t and t.startswith("#"):
                    chapter_title = re.sub(r"^#+\s*", "", t)
                    break
            i += 1
            continue

        # Artykuł
        m_art = re.match(r"^#{1,6}\s+Artykuł\s+(\d+)\s*$", line)
        if m_art:
            art_num = m_art.group(1)
            art_title = ""
            body_lines = []

            # Tytuł artykułu — następny niepusty nagłówek
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and lines[j].strip().startswith("#"):
                art_title = re.sub(r"^#+\s*", "", lines[j].strip())
                j += 1
            else:
                j = i + 1

            # Treść — do następnego artykułu lub rozdziału
            while j < len(lines):
                next_line = lines[j]
                if re.match(r"^#{1,6}\s+Artykuł\s+\d+\s*$", next_line):
                    break
                if re.match(r"^#{1,6}\s+ROZDZIAŁ\s+", next_line):
                    break
                # Pomiń separatory stron i nagłówki stron
                if re.match(r"^---$", next_line) or re.match(r"^#{1,6}\s+Strona\s+\d+", next_line):
                    j += 1
                    continue
                body_lines.append(next_line)
                j += 1

            body = "\n".join(body_lines).strip()
            # Usuń przypisy na końcu
            body = re.sub(r"\n+> \([\w⁰-⁹]+\).*$", "", body, flags=re.MULTILINE).strip()

            if len(body) < 10:
                i = j
                continue

            full_text = f"Artykuł {art_num} RODO"
            if art_title:
                full_text += f" — {art_title}"
            full_text += f"\n\n{body}"

            chunks = split_into_chunks(full_text)
            for ci, chunk in enumerate(chunks):
                documents.append({
                    "doc_type": "gdpr_article",
                    "article_num": art_num,
                    "article_title": art_title,
                    "chapter": chapter_id,
                    "chapter_title": chapter_title,
                    "content_text": chunk,
                    "source": RODO_SOURCE,
                    "source_url": RODO_URL,
                    "chunk_index": ci,
                    "chunk_total": len(chunks),
                })
            art_count += 1
            i = j
            continue

        i += 1

    arts    = sum(1 for d in documents if d["doc_type"] == "gdpr_article")
    recits  = sum(1 for d in documents if d["doc_type"] == "gdpr_recital")
    print(f"Znaleziono {art_count} artykułów → {arts} chunków")
    print(f"Znaleziono {recits} motywów")
    print(f"Łącznie dokumentów: {len(documents)}")
    return documents


# ─────────────────────────── INDEKSOWANIE ────────────────────────────────────

def get_embedder(model_name: str):
    from sentence_transformers import SentenceTransformer
    print(f"Ładowanie modelu: {model_name}")
    return SentenceTransformer(model_name, trust_remote_code=True)


def embed_batch(texts: List[str], embedder, batch_size: int = 32) -> List[List[float]]:
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        vecs = embedder.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_vecs.extend(vecs.tolist())
        print(f"  Embeddingi: {min(i + batch_size, len(texts))}/{len(texts)}", end="\r")
    print()
    return all_vecs


def index_documents(documents: List[Dict], qdrant_url: str,
                    collection_name: str, model_name: str):
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue

    client = QdrantClient(url=qdrant_url, timeout=60)

    collections = [c.name for c in client.get_collections().collections]
    if collection_name not in collections:
        print(f"BŁĄD: Kolekcja '{collection_name}' nie istnieje.")
        sys.exit(1)

    # Usuń stare dokumenty RODO
    for dtype in ("gdpr_article", "gdpr_recital"):
        client.delete(
            collection_name=collection_name,
            points_selector=Filter(must=[
                FieldCondition(key="doc_type", match=MatchValue(value=dtype))
            ]),
        )
        print(f"Usunięto stare dokumenty: {dtype}")

    # Embeddingi
    embedder = get_embedder(model_name)
    texts = [d["content_text"] for d in documents]
    print(f"Generowanie embeddingów dla {len(texts)} dokumentów...")
    vectors = embed_batch(texts, embedder)

    # Upsert w batchach
    batch_size = 64
    total = len(documents)
    for i in range(0, total, batch_size):
        batch_docs = documents[i:i + batch_size]
        batch_vecs = vectors[i:i + batch_size]
        points = [
            PointStruct(id=str(uuid.uuid4()), vector=vec, payload=doc)
            for doc, vec in zip(batch_docs, batch_vecs)
        ]
        client.upsert(collection_name=collection_name, points=points)
        print(f"Zaindeksowano: {min(i + batch_size, total)}/{total}", end="\r")
        time.sleep(0.05)

    arts   = sum(1 for d in documents if d["doc_type"] == "gdpr_article")
    recits = sum(1 for d in documents if d["doc_type"] == "gdpr_recital")
    print(f"\nGotowe! Zaindeksowano {total} dokumentów RODO.")
    print(f"  Artykuły: {arts}")
    print(f"  Motywy:   {recits}")


# ─────────────────────────── MAIN ────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Indeksuje RODO w Qdrant z pliku Markdown")
    parser.add_argument("--md",         default=DEFAULT_MD)
    parser.add_argument("--qdrant",     default="http://localhost:6333")
    parser.add_argument("--collection", default="uodo_decisions")
    parser.add_argument("--model",      default="sdadas/mmlw-retrieval-roberta-large")
    parser.add_argument("--dry-run",    action="store_true")
    args = parser.parse_args()

    md_path = Path(args.md)
    if not md_path.exists():
        print(f"BŁĄD: Plik '{md_path}' nie istnieje.")
        sys.exit(1)

    print(f"Wczytuję: {md_path}")
    text = md_path.read_text(encoding="utf-8")

    print("Parsowanie...")
    documents = parse_rodo_md(text)

    if args.dry_run:
        print("\n=== DRY RUN — przykłady ===")
        for doc in documents[:2]:
            print(f"\n[{doc['doc_type']}] {doc['article_num']} | rozdział {doc['chapter']}")
            print(doc["content_text"][:300])
        return

    index_documents(documents, args.qdrant, args.collection, args.model)


if __name__ == "__main__":
    main()
