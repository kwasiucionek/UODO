#!/usr/bin/env python3
"""
rodo_indexer.py — pobiera tekst RODO (2016/679) z EUR-Lex i indeksuje go w Qdrant.

Uruchomienie:
  python rodo_indexer.py
  python rodo_indexer.py --qdrant http://localhost:6333 --collection uodo_decisions

Opcje:
  --qdrant      URL Qdrant (domyślnie http://localhost:6333)
  --collection  Nazwa kolekcji (domyślnie uodo_decisions)
  --model       Model embeddingowy (domyślnie sdadas/mmlw-retrieval-roberta-large)
  --pdf         Ścieżka do lokalnego PDF (opcjonalne, zamiast pobierania)
  --dry-run     Parsuj i wypisz bez indeksowania
"""

import argparse
import re
import sys
import time
import unicodedata
import uuid
from pathlib import Path
from typing import List, Dict, Optional

# ─────────────────────────────────────────────────────────────────────────────

RODO_PDF_URL = "https://eur-lex.europa.eu/legal-content/PL/TXT/PDF/?uri=CELEX:32016R0679"
RODO_SOURCE  = "RODO 2016/679"
RODO_ISAP    = "https://eur-lex.europa.eu/legal-content/PL/TXT/?uri=CELEX:32016R0679"

CHUNK_MAX_CHARS = 1200   # max znaków na chunk artykułu
CHUNK_OVERLAP   = 100    # nakładanie chunków


# ─────────────────────────── POBIERANIE PDF ───────────────────────────────────

FALLBACK_URLS = [
    "https://eur-lex.europa.eu/legal-content/PL/TXT/PDF/?uri=CELEX:32016R0679",
    "https://eur-lex.europa.eu/legal-content/PL/TXT/PDF/?uri=OJ:L:2016:119:FULL",
]

MANUAL_INSTRUCTIONS = """
EUR-Lex blokuje automatyczne pobieranie. Pobierz PDF ręcznie:

  1. Otwórz: https://eur-lex.europa.eu/legal-content/PL/TXT/PDF/?uri=CELEX:32016R0679
  2. Zapisz jako: rodo_2016_679_pl.pdf
  3. Uruchom ponownie: python rodo_indexer.py

lub podaj ścieżkę do pobranego pliku:
  python rodo_indexer.py --pdf /sciezka/do/rodo.pdf
"""


def is_valid_pdf(path: Path) -> bool:
    """Sprawdza czy plik zaczyna się od magic bytes PDF."""
    try:
        with open(path, "rb") as f:
            return f.read(5) == b"%PDF-"
    except Exception:
        return False


def download_pdf(dest: Path) -> Path:
    import requests
    for url in FALLBACK_URLS:
        print(f"Próba pobierania: {url}")
        try:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
                ),
                "Accept": "application/pdf,*/*",
                "Referer": "https://eur-lex.europa.eu/",
            }
            r = requests.get(url, headers=headers, timeout=60, stream=True)
            r.raise_for_status()
            tmp = dest.with_suffix(".tmp")
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    f.write(chunk)
            if is_valid_pdf(tmp):
                tmp.rename(dest)
                size_kb = dest.stat().st_size // 1024
                print(f"Pobrano: {dest} ({size_kb} KB)")
                return dest
            else:
                tmp.unlink(missing_ok=True)
                print(f"  Odpowiedź nie jest PDF-em (prawdopodobnie blokada)")
        except Exception as e:
            print(f"  Błąd: {e}")

    print(MANUAL_INSTRUCTIONS)
    sys.exit(1)


# ─────────────────────────── EKSTRAKCJA TEKSTU ────────────────────────────────

def extract_text_from_pdf(pdf_path: Path) -> str:
    """Wyciąga surowy tekst z PDF za pomocą pdfminer.six."""
    try:
        from pdfminer.high_level import extract_text
        text = extract_text(str(pdf_path))
        print(f"Wyekstrahowano {len(text)} znaków z PDF")
        return text
    except ImportError:
        print("Brak pdfminer.six — instaluję...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "pdfminer.six", "-q"], check=True)
        from pdfminer.high_level import extract_text
        return extract_text(str(pdf_path))


def normalize(text: str) -> str:
    """Normalizuje znaki Unicode, usuwa śmieci z PDF."""
    text = unicodedata.normalize("NFKC", text)
    # Usuń dzielenie wyrazów (łącznik na końcu linii)
    text = re.sub(r"-\n(\w)", r"\1", text)
    # Scal linie wewnątrz akapitu
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    # Normalizuj wielokrotne spacje
    text = re.sub(r" {2,}", " ", text)
    # Normalizuj wielokrotne nowe linie
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ─────────────────────────── PARSOWANIE ──────────────────────────────────────

# Wzorce dla struktury RODO
_RE_RECITAL = re.compile(
    r"(?:^|\n)\s*\((\d{1,3})\)\s+(.+?)(?=\n\s*\(\d{1,3}\)|\nROZDZIAŁ\s+[IVX]+|\nArtykuł\s+\d)",
    re.DOTALL,
)
_RE_ARTICLE = re.compile(
    r"\nArtykuł\s+(\d+)\s*\n([^\n]+)\n(.*?)(?=\nArtykuł\s+\d+\s*\n|\Z)",
    re.DOTALL,
)
_RE_CHAPTER = re.compile(
    r"\nROZDZIAŁ\s+([IVX]+)\s*\n([^\n]+)",
)


def parse_chapters(text: str) -> Dict[str, str]:
    """Mapuje numer artykułu → tytuł rozdziału."""
    chapters = {}
    chapter_id, chapter_title = "", ""
    # Przeskanuj tekst linia po linii
    lines = text.split("\n")
    current_chapter = ("", "")
    article_to_chapter = {}

    for i, line in enumerate(lines):
        m_ch = re.match(r"\s*ROZDZIAŁ\s+([IVX]+)\s*$", line.strip())
        if m_ch:
            chapter_id = m_ch.group(1)
            # Tytuł rozdziału to zazwyczaj kolejna niepusta linia
            for j in range(i + 1, min(i + 4, len(lines))):
                t = lines[j].strip()
                if t and not re.match(r"Artykuł\s+\d+", t):
                    chapter_title = t
                    break
            current_chapter = (chapter_id, chapter_title)
            continue
        m_art = re.match(r"\s*Artykuł\s+(\d+)\s*$", line.strip())
        if m_art:
            article_to_chapter[m_art.group(1)] = current_chapter

    return article_to_chapter


def split_into_chunks(text: str, max_chars: int, overlap: int) -> List[str]:
    """Dzieli długi tekst na nakładające się chunki po akapitach."""
    if len(text) <= max_chars:
        return [text]

    paragraphs = re.split(r"\n\n+", text)
    chunks, current, current_len = [], [], 0

    for para in paragraphs:
        para_len = len(para)
        if current_len + para_len > max_chars and current:
            chunks.append("\n\n".join(current))
            # Zachowaj ostatni akapit jako overlap
            overlap_paras = []
            overlap_len = 0
            for p in reversed(current):
                if overlap_len + len(p) <= overlap:
                    overlap_paras.insert(0, p)
                    overlap_len += len(p)
                else:
                    break
            current = overlap_paras
            current_len = overlap_len
        current.append(para)
        current_len += para_len

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def parse_rodo(text: str) -> List[Dict]:
    """
    Parsuje tekst RODO i zwraca listę dokumentów do zaindeksowania.
    Każdy dokument to artykuł lub motyw.
    """
    text = normalize(text)
    art_to_chapter = parse_chapters(text)
    documents = []

    # ── Motywy ────────────────────────────────────────────────────
    # Motywy są przed rozdziałem I, identyfikowane przez (N) na początku
    recital_section_match = re.search(
        r"mając na uwadze, co następuje:(.*?)PRZYJMUJĄ NINIEJSZE ROZPORZĄDZENIE",
        text, re.DOTALL
    )
    if recital_section_match:
        recital_text = recital_section_match.group(1)
        recitals = re.findall(
            r"\((\d{1,3})\)\s+(.*?)(?=\n\(\d{1,3}\)|\Z)",
            recital_text, re.DOTALL
        )
        for num, content in recitals:
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
                "source_url": RODO_ISAP,
                "chunk_index": 0,
                "chunk_total": 1,
            })
        print(f"Znaleziono {len(recitals)} motywów")
    else:
        print("UWAGA: Nie znaleziono sekcji motywów")

    # ── Artykuły ──────────────────────────────────────────────────
    articles = re.findall(
        r"Artykuł\s+(\d+)\s*\n([^\n]+)\n(.*?)(?=Artykuł\s+\d+\s*\n|\Z)",
        text, re.DOTALL
    )

    if not articles:
        # Fallback: bardziej liberalny wzorzec
        articles = re.findall(
            r"Artykuł\s+(\d+)\s*\n(.*?)(?=Artykuł\s+\d+|\Z)",
            text, re.DOTALL
        )
        articles = [(num, "", body) for num, body in articles]

    art_count = 0
    for num, title, body in articles:
        title = title.strip()
        body = body.strip()
        if not body or len(body) < 10:
            continue

        chapter_id, chapter_title = art_to_chapter.get(num, ("", ""))
        full_text = f"Artykuł {num} RODO"
        if title:
            full_text += f" — {title}"
        full_text += f"\n\n{body}"

        chunks = split_into_chunks(full_text, CHUNK_MAX_CHARS, CHUNK_OVERLAP)
        for i, chunk in enumerate(chunks):
            documents.append({
                "doc_type": "gdpr_article",
                "article_num": num,
                "article_title": title,
                "chapter": chapter_id,
                "chapter_title": chapter_title,
                "content_text": chunk,
                "source": RODO_SOURCE,
                "source_url": RODO_ISAP,
                "chunk_index": i,
                "chunk_total": len(chunks),
            })
        art_count += 1

    print(f"Znaleziono {art_count} artykułów → {sum(1 for d in documents if d['doc_type'] == 'gdpr_article')} chunków")
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
    from qdrant_client.models import (
        PointStruct, VectorParams, Distance,
        FieldCondition, Filter, MatchValue
    )

    client = QdrantClient(url=qdrant_url, timeout=60)

    # Sprawdź czy kolekcja istnieje
    collections = [c.name for c in client.get_collections().collections]
    if collection_name not in collections:
        print(f"BŁĄD: Kolekcja '{collection_name}' nie istnieje. Najpierw uruchom uodo_indexer.py")
        sys.exit(1)

    # Usuń istniejące dokumenty RODO z kolekcji
    existing_types = ["gdpr_article", "gdpr_recital"]
    for dtype in existing_types:
        deleted = client.delete(
            collection_name=collection_name,
            points_selector=Filter(must=[
                FieldCondition(key="doc_type", match=MatchValue(value=dtype))
            ]),
        )
        print(f"Usunięto stare dokumenty typu '{dtype}'")

    # Embeddingi
    embedder = get_embedder(model_name)
    texts = [d["content_text"] for d in documents]
    print(f"Generowanie embeddingów dla {len(texts)} dokumentów...")
    vectors = embed_batch(texts, embedder)

    # Indeksowanie w batchach
    batch_size = 64
    total = len(documents)
    for i in range(0, total, batch_size):
        batch_docs = documents[i:i + batch_size]
        batch_vecs = vectors[i:i + batch_size]
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload=doc,
            )
            for doc, vec in zip(batch_docs, batch_vecs)
        ]
        client.upsert(collection_name=collection_name, points=points)
        print(f"Zaindeksowano: {min(i + batch_size, total)}/{total}", end="\r")
        time.sleep(0.05)

    print(f"\nGotowe! Zaindeksowano {total} dokumentów RODO do kolekcji '{collection_name}'")

    # Podsumowanie
    arts    = sum(1 for d in documents if d["doc_type"] == "gdpr_article")
    recitals = sum(1 for d in documents if d["doc_type"] == "gdpr_recital")
    print(f"  Artykuły:  {arts}")
    print(f"  Motywy:    {recitals}")


# ─────────────────────────── MAIN ────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Indeksuje RODO w Qdrant")
    parser.add_argument("--qdrant",     default="http://localhost:6333")
    parser.add_argument("--collection", default="uodo_decisions")
    parser.add_argument("--model",      default="sdadas/mmlw-retrieval-roberta-large")
    parser.add_argument("--pdf",        default=None, help="Ścieżka do lokalnego PDF")
    parser.add_argument("--dry-run",    action="store_true", help="Tylko parsuj, nie indeksuj")
    args = parser.parse_args()

    # 1. PDF
    DEFAULT_PDF = Path("rodo_2016_679_pl.pdf")
    if args.pdf:
        pdf_path = Path(args.pdf)
        if not pdf_path.exists():
            print(f"BŁĄD: Plik '{pdf_path}' nie istnieje.")
            # Sprawdź czy może jest domyślny plik
            if DEFAULT_PDF.exists():
                print(f"Znaleziono domyślny plik: {DEFAULT_PDF} — używam go.")
                pdf_path = DEFAULT_PDF
            else:
                print(MANUAL_INSTRUCTIONS)
                sys.exit(1)
    else:
        pdf_path = DEFAULT_PDF
        if not pdf_path.exists():
            download_pdf(pdf_path)
        else:
            print(f"Używam istniejącego PDF: {pdf_path}")

    if not is_valid_pdf(pdf_path):
        print(f"BŁĄD: '{pdf_path}' nie jest poprawnym plikiem PDF (może to HTML z blokady).")
        print(MANUAL_INSTRUCTIONS)
        sys.exit(1)

    # 2. Ekstrakcja tekstu
    print("Ekstraktuję tekst z PDF...")
    raw_text = extract_text_from_pdf(pdf_path)

    # 3. Parsowanie
    print("Parsowanie artykułów i motywów...")
    documents = parse_rodo(raw_text)
    print(f"Łącznie dokumentów do zaindeksowania: {len(documents)}")

    if args.dry_run:
        print("\n=== DRY RUN — pierwsze 3 dokumenty ===")
        for doc in documents[:3]:
            print(f"\n[{doc['doc_type']}] {doc['article_num']}")
            print(f"Rozdział: {doc['chapter']} — {doc['chapter_title']}")
            print(f"Tekst ({len(doc['content_text'])} znaków):")
            print(doc["content_text"][:300])
            print("...")
        return

    # 4. Indeksowanie
    index_documents(documents, args.qdrant, args.collection, args.model)


if __name__ == "__main__":
    main()
