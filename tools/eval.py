#!/usr/bin/env python3
"""
eval.py — Automatyczna ewaluacja systemu UODO RAG.

Wzorzec z kursu Software 3.0 (lekcja 5.2–5.3): binarny leaderboard
zamiast subiektywnych ocen 1–10. Każdy test zwraca 0 lub 1.

Uruchomienie:
    python eval.py                     # pełna ewaluacja
    python eval.py --question 3        # tylko pytanie nr 3
    python eval.py --verbose           # z pełnymi odpowiedziami

Wymagania:
    pip install qdrant-client sentence-transformers groq requests python-dotenv
"""

import argparse
import json
import os
import sys
import time
from typing import Any

# ─────────────────────────── KONFIGURACJA ────────────────────────

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

QDRANT_URL      = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "uodo_decisions"
EMBED_MODEL     = os.getenv("EMBED_MODEL", "sdadas/mmlw-retrieval-roberta-large")
GROQ_API_KEY    = os.getenv("GROQ_API_KEY", "")
DEFAULT_MODEL   = "llama-3.3-70b-versatile"

# ─────────────────────────── ZŁOTE PYTANIA ────────────────────────
# 10 pytań ze zdefiniowanymi kryteriami sukcesu.
# Każde kryterium to binarne sprawdzenie (0/1) na odpowiedzi LLM.
#
# Format każdego testu:
#   question:    pytanie użytkownika
#   description: co sprawdzamy
#   checks:      lista funkcji lambda(answer: str) -> bool
#   check_names: nazwy sprawdzeń (1:1 z checks)

GOLDEN_QUESTIONS: list[dict[str, Any]] = [
    {
        "id": "GQ-001",
        "question": "Jakie kary może nałożyć Prezes UODO?",
        "description": "Odpowiedź powinna wspomnieć o karach administracyjnych i ich wysokości",
        "checks": [
            lambda a: any(w in a.lower() for w in ["kara", "administracyjna", "pieniężna"]),
            lambda a: any(w in a.lower() for w in ["dkn", "uodo", "prezes"]),
            lambda a: "art" in a.lower() or "artykuł" in a.lower(),
        ],
        "check_names": [
            "Wspomina o karze administracyjnej",
            "Cytuje sygnaturę lub organ (UODO/Prezes)",
            "Powołuje się na artykuł",
        ],
    },
    {
        "id": "GQ-002",
        "question": "Kiedy wymagane jest zgłoszenie naruszenia danych do UODO?",
        "description": "Odpowiedź powinna zawierać termin 72h i Art. 33 RODO",
        "checks": [
            lambda a: "72" in a,
            lambda a: "art" in a.lower() and "33" in a,
            lambda a: any(w in a.lower() for w in ["naruszenie", "zgłoszenie", "incydent"]),
        ],
        "check_names": [
            "Podaje termin 72 godzin",
            "Cytuje Art. 33 RODO",
            "Używa pojęcia 'naruszenie' lub 'zgłoszenie'",
        ],
    },
    {
        "id": "GQ-003",
        "question": "Co to jest podstawa prawna przetwarzania danych osobowych?",
        "description": "Odpowiedź powinna wymienić podstawy z Art. 6 RODO",
        "checks": [
            lambda a: "art" in a.lower() and "6" in a,
            lambda a: any(w in a.lower() for w in ["zgoda", "umowa", "obowiązek", "interes"]),
            lambda a: "rodo" in a.lower() or "2016/679" in a,
        ],
        "check_names": [
            "Cytuje Art. 6 RODO",
            "Wymienia co najmniej jedną podstawę (zgoda/umowa/obowiązek/interes)",
            "Odwołuje się do RODO",
        ],
    },
    {
        "id": "GQ-004",
        "question": "Jakie obowiązki ma administrator danych wobec osoby, której dane dotyczą?",
        "description": "Odpowiedź powinna wymienić obowiązek informacyjny",
        "checks": [
            lambda a: any(w in a.lower() for w in ["informacyjny", "obowiązek informacyjny", "art. 13", "art. 14"]),
            lambda a: any(w in a.lower() for w in ["administrator", "podmiot"]),
            lambda a: "art" in a.lower(),
        ],
        "check_names": [
            "Wspomina o obowiązku informacyjnym lub Art. 13/14",
            "Wymienia administratora jako podmiot zobowiązany",
            "Powołuje się na artykuł",
        ],
    },
    {
        "id": "GQ-005",
        "question": "Czym jest inspektor ochrony danych i kiedy trzeba go wyznaczyć?",
        "description": "Odpowiedź powinna wyjaśnić rolę IOD i warunki obowiązku wyznaczenia",
        "checks": [
            lambda a: any(w in a.lower() for w in ["inspektor", "iod", "dpo"]),
            lambda a: any(w in a.lower() for w in ["wyznaczenie", "obowiązek", "musi"]),
            lambda a: "art" in a.lower() and ("37" in a or "38" in a or "39" in a),
        ],
        "check_names": [
            "Wyjaśnia pojęcie IOD/DPO",
            "Opisuje obowiązek wyznaczenia",
            "Cytuje Art. 37, 38 lub 39 RODO",
        ],
    },
    {
        "id": "GQ-006",
        "question": "Jakie prawa przysługują osobie, której dane są przetwarzane?",
        "description": "Odpowiedź powinna wymienić prawa: dostęp, sprostowanie, usunięcie, przenoszenie",
        "checks": [
            lambda a: any(w in a.lower() for w in ["dostęp", "wgląd"]),
            lambda a: any(w in a.lower() for w in ["usunięcie", "zapomnienie", "prawo do bycia zapomnianym"]),
            lambda a: any(w in a.lower() for w in ["sprostowanie", "poprawienie"]),
        ],
        "check_names": [
            "Wymienia prawo dostępu",
            "Wymienia prawo do usunięcia danych",
            "Wymienia prawo do sprostowania",
        ],
    },
    {
        "id": "GQ-007",
        "question": "Co to jest umowa powierzenia przetwarzania danych?",
        "description": "Odpowiedź powinna wyjaśnić umowę powierzenia i Art. 28 RODO",
        "checks": [
            lambda a: any(w in a.lower() for w in ["powierzenie", "procesor", "podmiot przetwarzający"]),
            lambda a: "art" in a.lower() and "28" in a,
            lambda a: any(w in a.lower() for w in ["umowa", "kontrakt"]),
        ],
        "check_names": [
            "Wyjaśnia pojęcie powierzenia/procesora",
            "Cytuje Art. 28 RODO",
            "Wspomina o umowie",
        ],
    },
    {
        "id": "GQ-008",
        "question": "Jakie dane uznaje się za szczególne kategorie danych osobowych?",
        "description": "Odpowiedź powinna wymienić przykłady danych wrażliwych z Art. 9 RODO",
        "checks": [
            lambda a: any(w in a.lower() for w in ["szczególne", "wrażliwe", "art. 9", "art 9"]),
            lambda a: any(w in a.lower() for w in ["zdrowie", "genetyczne", "rasowe", "biometryczne", "wyznanie"]),
            lambda a: "rodo" in a.lower() or "art" in a.lower(),
        ],
        "check_names": [
            "Używa pojęcia 'szczególne kategorie' lub Art. 9",
            "Wymienia co najmniej jeden przykład (zdrowie/genetyczne/rasowe itp.)",
            "Odwołuje się do RODO lub artykułu",
        ],
    },
    {
        "id": "GQ-009",
        "question": "Kiedy można przekazywać dane osobowe do krajów trzecich?",
        "description": "Odpowiedź powinna opisać mechanizmy transferu (rozdział V RODO)",
        "checks": [
            lambda a: any(w in a.lower() for w in ["kraj trzeci", "transfer", "przekazanie"]),
            lambda a: any(w in a.lower() for w in ["adequacy", "adequateness", "odpowiedni stopień ochrony", "standardowe klauzule", "bcr", "wiążące reguły"]),
            lambda a: "art" in a.lower() or "rozdział v" in a.lower(),
        ],
        "check_names": [
            "Wspomina o przekazaniu do krajów trzecich",
            "Wymienia mechanizm transferu (decyzja o adekwatności / klauzule / BCR)",
            "Powołuje się na artykuł lub rozdział V RODO",
        ],
    },
    {
        "id": "GQ-010",
        "question": "Co to jest minimalizacja danych i zasada ograniczenia celu?",
        "description": "Odpowiedź powinna opisać zasady z Art. 5 RODO",
        "checks": [
            lambda a: "minimalizacja" in a.lower() or "minimalizację" in a.lower(),
            lambda a: any(w in a.lower() for w in ["cel", "ograniczenie celu", "purpose limitation"]),
            lambda a: "art" in a.lower() and "5" in a,
        ],
        "check_names": [
            "Wyjaśnia zasadę minimalizacji danych",
            "Opisuje ograniczenie celu przetwarzania",
            "Cytuje Art. 5 RODO",
        ],
    },
]

# ─────────────────────────── FUNKCJE POMOCNICZE ──────────────────


def get_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL)


def semantic_search(query: str, top_k: int = 8) -> list[dict[str, Any]]:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, FieldCondition, MatchAny
    client = QdrantClient(url=QDRANT_URL)
    embedder = get_embedder()
    vec = embedder.encode(query).tolist()
    qdrant_filter = Filter(must=[
        FieldCondition(key="doc_type", match=MatchAny(any=[
            "uodo_decision", "legal_act_article", "gdpr_article", "gdpr_recital"
        ]))
    ])
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=vec,
        limit=top_k,
        query_filter=qdrant_filter,
        with_payload=True,
    )
    docs = []
    for r in results:
        d = dict(r.payload or {})
        d["_score"] = r.score
        docs.append(d)
    return docs


def build_simple_context(docs: list[dict[str, Any]], query: str, max_chars: int = 10000) -> str:
    parts = [f"Pytanie: {query}\n\nDokumenty:\n"]
    chars = len(parts[0])
    for i, doc in enumerate(docs, 1):
        dtype = doc.get("doc_type", "")
        if dtype == "uodo_decision":
            sig = doc.get("signature", "?")
            text = doc.get("content_text", "")[:600]
            block = f"[{i}] DECYZJA {sig}\n{text}\n"
        elif dtype == "legal_act_article":
            art = doc.get("article_num", "?")
            text = doc.get("content_text", "")[:600]
            block = f"[{i}] Art. {art} u.o.d.o.\n{text}\n"
        else:
            art = doc.get("article_num", "?")
            text = doc.get("content_text", "")[:600]
            block = f"[{i}] Art. {art} RODO\n{text}\n"
        if chars + len(block) > max_chars:
            break
        parts.append(block)
        chars += len(block)
    return "\n---\n".join(parts)


def call_llm(query: str, context: str) -> str:
    """Wywołuje LLM i zwraca pełną odpowiedź (bez streamowania)."""
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)
    system = (
        "Jesteś ekspertem prawa ochrony danych osobowych. "
        "Odpowiadaj WYŁĄCZNIE na podstawie podanych dokumentów. "
        "Cytuj sygnatury decyzji i numery artykułów. "
        "Odpowiadaj po polsku."
    )
    resp = client.chat.completions.create(  # type: ignore[call-overload]
        model=DEFAULT_MODEL,
        messages=[  # type: ignore[arg-type]
            {"role": "system", "content": system},
            {"role": "user", "content": f"Pytanie: {query}\n\nDokumenty:\n{context}"},
        ],
        max_tokens=1024,
        temperature=0.0,
    )
    return resp.choices[0].message.content or ""


# ─────────────────────────── RUNNER ──────────────────────────────


def run_single(gq: dict[str, Any], verbose: bool = False) -> dict[str, Any]:
    """Uruchamia jeden test i zwraca wyniki."""
    print(f"\n{'='*60}")
    print(f"  {gq['id']}: {gq['question']}")
    print(f"  {gq['description']}")
    print(f"{'='*60}")

    t0 = time.time()
    try:
        docs = semantic_search(gq["question"])
        context = build_simple_context(docs, gq["question"])
        answer = call_llm(gq["question"], context)
    except Exception as e:
        print(f"  ❌ BŁĄD: {e}")
        return {"id": gq["id"], "question": gq["question"], "error": str(e), "checks": [], "passed": 0, "total": len(gq["checks"])}

    elapsed = time.time() - t0

    if verbose:
        print(f"\n  ODPOWIEDŹ:\n  {answer[:500]}{'...' if len(answer) > 500 else ''}\n")

    check_results = []
    for name, check_fn in zip(gq["check_names"], gq["checks"]):
        try:
            passed = bool(check_fn(answer))
        except Exception:
            passed = False
        icon = "✅" if passed else "❌"
        print(f"  {icon} {name}")
        check_results.append({"name": name, "passed": passed})

    total = len(gq["checks"])
    passed_count = sum(1 for c in check_results if c["passed"])
    print(f"\n  Wynik: {passed_count}/{total} ({elapsed:.1f}s)")

    return {
        "id": gq["id"],
        "question": gq["question"],
        "checks": check_results,
        "passed": passed_count,
        "total": total,
        "elapsed_s": round(elapsed, 1),
    }


def run_all(question_idx: int | None = None, verbose: bool = False) -> None:
    questions = GOLDEN_QUESTIONS
    if question_idx is not None:
        idx = question_idx - 1
        if not (0 <= idx < len(questions)):
            print(f"Nieprawidłowy numer pytania: {question_idx} (dostępne: 1–{len(questions)})")
            sys.exit(1)
        questions = [questions[idx]]

    print(f"\n{'#'*60}")
    print(f"  UODO RAG — Ewaluacja ({len(questions)} pytań)")
    print(f"  Model: {DEFAULT_MODEL}")
    print(f"  Qdrant: {QDRANT_URL}/{COLLECTION_NAME}")
    print(f"{'#'*60}")

    all_results = []
    for gq in questions:
        result = run_single(gq, verbose=verbose)
        all_results.append(result)

    # Podsumowanie
    total_checks = sum(r["total"] for r in all_results)
    total_passed = sum(r["passed"] for r in all_results)
    total_questions = len(all_results)
    perfect = sum(1 for r in all_results if r["passed"] == r["total"])

    print(f"\n{'#'*60}")
    print(f"  PODSUMOWANIE LEADERBOARDU")
    print(f"{'#'*60}")
    print(f"  Pytania: {perfect}/{total_questions} w pełni zdanych")
    print(f"  Sprawdzenia: {total_passed}/{total_checks} zdanych")
    pct = total_passed / total_checks * 100 if total_checks else 0
    print(f"  Wynik ogólny: {pct:.1f}%")

    # Tabela wyników
    print(f"\n  {'ID':<10} {'Zdanych':<10} {'Wynik'}")
    print(f"  {'-'*40}")
    for r in all_results:
        bar = "█" * r["passed"] + "░" * (r["total"] - r["passed"])
        icon = "✅" if r["passed"] == r["total"] else ("⚠️" if r["passed"] > 0 else "❌")
        print(f"  {r['id']:<10} {r['passed']}/{r['total']:<8} {icon} {bar}")

    # Zapisz wyniki do JSON
    output_path = "eval_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": DEFAULT_MODEL,
            "summary": {
                "questions_perfect": perfect,
                "questions_total": total_questions,
                "checks_passed": total_passed,
                "checks_total": total_checks,
                "score_pct": round(pct, 1),
            },
            "results": all_results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n  Wyniki zapisane: {output_path}")


# ─────────────────────────── MAIN ────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UODO RAG — Ewaluacja binarnym leaderboardem")
    parser.add_argument("--question", type=int, default=None, help="Numer pytania (1–10)")
    parser.add_argument("--verbose", action="store_true", help="Wyświetl pełne odpowiedzi")
    args = parser.parse_args()

    if not GROQ_API_KEY:
        print("❌ Brak GROQ_API_KEY — ustaw w .env lub zmiennej środowiskowej")
        sys.exit(1)

    run_all(question_idx=args.question, verbose=args.verbose)
