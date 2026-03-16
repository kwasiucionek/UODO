#!/usr/bin/env python3
"""
enrich_jsonl_taxonomy.py — wzbogaca istniejący JSONL o pola taksonomii
z already pobranego pola meta.terms, bez ponownego scrapowania.

Użycie:
  python enrich_jsonl_taxonomy.py
  python enrich_jsonl_taxonomy.py --input uodo_decisions.jsonl --output uodo_decisions_enriched.jsonl
"""
import argparse
import json
from pathlib import Path


def parse_terms_taxonomy(terms: list) -> dict:
    result = {
        "term_decision_type": [],
        "term_violation_type": [],
        "term_legal_basis": [],
        "term_corrective_measure": [],
        "term_sector": [],
    }
    for term in (terms or []):
        if not isinstance(term, dict):
            continue
        label = term.get("label", "")
        name = term.get("name", {})
        name_pl = name.get("pl", "") if isinstance(name, dict) else str(name)
        if not label or not name_pl:
            continue
        prefix = label.split(".")[0]
        if prefix == "1":
            result["term_decision_type"].append(name_pl)
        elif prefix == "2":
            result["term_violation_type"].append(name_pl)
        elif prefix == "3":
            result["term_legal_basis"].append(name_pl)
        elif prefix == "4":
            result["term_corrective_measure"].append(name_pl)
        elif prefix == "9":
            result["term_sector"].append(name_pl)
    return result


def enrich(input_path: str, output_path: str):
    docs = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))

    print(f"Wczytuję: {len(docs)} dokumentów z {input_path}")

    enriched = 0
    for doc in docs:
        meta = doc.get("meta", {})
        terms = meta.get("terms", [])
        taxonomy = parse_terms_taxonomy(terms)

        # Dodaj pola tylko jeśli nie istnieją lub są puste
        updated = False
        for field, values in taxonomy.items():
            if values and not doc.get(field):
                doc[field] = values
                updated = True

        if updated:
            enriched += 1

    print(f"Wzbogacono: {enriched}/{len(docs)} dokumentów")

    with open(output_path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"Zapisano: {output_path}")

    # Pokaż przykład
    for doc in docs:
        if doc.get("term_decision_type"):
            print(f"\nPrzykład — {doc['signature']}:")
            print(f"  term_decision_type:      {doc['term_decision_type']}")
            print(f"  term_violation_type:     {doc['term_violation_type'][:3]}")
            print(f"  term_legal_basis:        {doc['term_legal_basis'][:3]}")
            print(f"  term_corrective_measure: {doc['term_corrective_measure']}")
            print(f"  term_sector:             {doc['term_sector'][:3]}")
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="uodo_decisions.jsonl")
    parser.add_argument("--output", default="uodo_decisions_enriched.jsonl")
    args = parser.parse_args()
    enrich(args.input, args.output)


if __name__ == "__main__":
    main()
