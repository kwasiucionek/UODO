#!/usr/bin/env python3
"""
UODO API Scraper — pobiera orzeczenia UODO przez REST API portalu.

Nie wymaga przeglądarki headless — tylko: pip install requests

Użycie:
  python uodo_scraper.py --test
  python uodo_scraper.py --output uodo_decisions.jsonl
  python uodo_scraper.py --output uodo_decisions.jsonl --user login --password haslo
  python uodo_scraper.py --output uodo_decisions.jsonl --date-from 2024-01-01

API base: https://orzeczenia.uodo.gov.pl/api
"""

import argparse
import json
import os
import re
import time
from typing import Dict, List, Optional

import requests
from requests.auth import HTTPBasicAuth

# ─────────────────────────── CONFIG ──────────────────────────────

API_BASE = "https://orzeczenia.uodo.gov.pl/api"
DEFAULT_DELAY = 0.3
BATCH_SIZE = 100
MAX_RETRIES = 3
TIMEOUT = 30

# Pola z indeksu PublicDocument (wg indexes.yml + openapi.yml example)
# Dostępne: id, time, mtime, kind, refid, refname,
#   publicator_country, publicator_type, publicator_subtype, publicator_year,
#   date_announcement, date_publication, keywords, title_pl, content_pl
SEARCH_FIELDS = "id,refid,refname,keywords,title_pl,date_announcement,date_publication"


# ─────────────────────────── HTTP ────────────────────────────────

def make_session(user: str = None, password: str = None) -> requests.Session:
    s = requests.Session()
    if user and password:
        s.auth = HTTPBasicAuth(user, password)
    s.headers["Accept"] = "application/json"
    return s


def get(session: requests.Session, url: str,
        retries: int = MAX_RETRIES, accept: str = None) -> Optional[requests.Response]:
    headers = {"Accept": accept} if accept else {}
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=TIMEOUT, headers=headers)
            if r.status_code == 200:
                return r
            if r.status_code == 404:
                return None
            if r.status_code == 401:
                print("  ❌ HTTP 401 — wymagana autoryzacja (--user / --password)")
                return None
            print(f"  ⚠️ HTTP {r.status_code} dla {url} (próba {attempt+1})")
        except Exception as e:
            print(f"  ⚠️ Błąd połączenia: {e} (próba {attempt+1})")
        if attempt < retries - 1:
            time.sleep(2)
    return None


# ─────────────────────────── SEARCH ──────────────────────────────

def fetch_document_list(session: requests.Session,
                        date_from: str = None,
                        date_to: str = None) -> List[Dict]:
    """
    Pobiera listę wszystkich dokumentów UODO z kluczowymi polami indeksu.
    Jeden batch request zwraca keywords, title_pl, daty — bez osobnych żądań per dokument.
    """
    date_from = date_from or ""
    date_to = date_to or ""
    timespan = f"{date_from},{date_to}"

    all_docs = []
    offset = 0

    print(f"🔍 Pobieranie listy dokumentów UODO...")
    print(f"   Zakres dat: '{timespan}' (puste = wszystkie)")

    while True:
        url = (
            f"{API_BASE}/documents/search/PublicDocument/{timespan}"
            f"/publicator_subtype:eq:uodo"
            f"?from={offset}&count={BATCH_SIZE}&order=-id&fields={SEARCH_FIELDS}"
        )
        r = get(session, url)
        if not r:
            print(f"  ❌ Błąd pobierania przy offset={offset}")
            break

        batch = r.json()
        if not batch:
            break

        all_docs.extend(batch)
        offset += len(batch)
        print(f"  📋 {offset} dokumentów...")

        if len(batch) < BATCH_SIZE:
            break

    print(f"✅ Łącznie: {len(all_docs)} dokumentów")
    return all_docs


# ─────────────────────────── KONWERSJE ───────────────────────────

def refid_to_signature(refid: str) -> str:
    """
    urn:ndoc:gov:pl:uodo:2025:dkn_5131_9  → DKN.5131.9.2025
    urn:ndoc:gov:pl:uodo:2024:dkn_5130_2  → DKN.5130.2.2024
    """
    m = re.search(r"uodo:(\d{4}):([\w]+)$", refid)
    if not m:
        return refid
    year = m.group(1)
    code = m.group(2).upper().replace("_", ".")
    # Usuń rok jeśli pojawia się na końcu kodu (np. DKN.5130.2.2024 → nie duplikuj)
    parts = [p for p in code.split(".") if p != year]
    if len(parts) >= 3:
        return f"{parts[0]}.{parts[1]}.{parts[2]}.{year}"
    return f"{code}.{year}"


def multilang_str(field) -> str:
    """Wyciąga string z pola {pl, en} lub zwraca as-is jeśli to już string."""
    if isinstance(field, dict):
        return field.get("pl") or field.get("en") or ""
    return str(field) if field else ""


# ─────────────────────────── PARSOWANIE meta.json ────────────────

def parse_meta(data: Dict) -> Dict:
    """
    Parsuje meta.json.

    Na podstawie metryki portalu pola to:
      name            — "Decyzja Prezesa UODO nr DKN.5131.12.2025"
      title           — pełny tytuł (opis naruszenia)
      status          — "nieprawomocna" / "prawomocna" (bezpośrednie pole!)
      terms           — [{name.{pl,en}, label, base}]  keywords
      entities        — [{title, name, function}]
      kind            — "decision" / "judgment"
      date_announcement — "2026-02-13" (Data ogłoszenia)
      date_publication  — "2026-03-02" (Data publikacji)
      publication.status — workflow: nonfinal/final
    """
    result = {
        "name": "",
        "title_full": "",
        "keywords": "",
        "keywords_list": [],
        "entities": [],
        "kind": "",
        "legal_status": "",       # prawomocna / nieprawomocna
        "pub_workflow_status": "",
        "date_issued": "",
        "date_published": "",
        # Taksonomia UODO — rozbita wg grup labelów
        "term_decision_type": [],    # 1.1x.xxx — Rodzaj decyzji
        "term_violation_type": [],   # 2.1x.xxx — Rodzaj naruszenia
        "term_legal_basis": [],      # 3.xx.xxx — Podstawa prawna (Rozporządzenie)
        "term_corrective_measure": [], # 4.1x.xxx — Środek naprawczy
        "term_sector": [],           # 9.1x.xxx — Sektor / Tematyka
    }

    if not data:
        return result

    result["name"] = multilang_str(data.get("name", {}))
    result["title_full"] = multilang_str(data.get("title", {}))

    # Status prawny — bezpośrednie pole "status" w meta.json
    result["legal_status"] = data.get("status", "")

    # Daty z meta.json — są w polu dates[] (ten sam format co dates.json)
    dates_parsed = parse_dates(data.get("dates", []))
    result["date_issued"] = dates_parsed["date_issued"]
    result["date_published"] = dates_parsed["date_published"]

    # refs z meta.json — API refs.json zwraca puste
    refs_parsed = parse_refs(data.get("refs", []))
    result["refs"] = refs_parsed

    # terms → keywords + taksonomia
    kw_names = []
    for term in (data.get("terms", []) or []):
        if not isinstance(term, dict):
            continue
        name = multilang_str(term.get("name", {}))
        label = term.get("label", "")
        if name:
            kw_names.append(name)
        if label:
            prefix = label.split(".")[0]
            if prefix == "1":
                result["term_decision_type"].append(name)
            elif prefix == "2":
                result["term_violation_type"].append(name)
            elif prefix == "3":
                result["term_legal_basis"].append(name)
            elif prefix == "4":
                result["term_corrective_measure"].append(name)
            elif prefix == "9":
                result["term_sector"].append(name)
    result["keywords_list"] = kw_names
    result["keywords"] = ", ".join(kw_names)

    # entities
    for ent in (data.get("entities", []) or []):
        if isinstance(ent, dict):
            result["entities"].append({
                "title": multilang_str(ent.get("title", {})),
                "name": multilang_str(ent.get("name", {})),
                "function": ent.get("function", "other"),
            })

    result["kind"] = data.get("kind", "")

    pub = data.get("publication", {})
    result["pub_workflow_status"] = (
        pub.get("status", "") if isinstance(pub, dict) else ""
    )

    return result


# ─────────────────────────── PARSOWANIE dates.json ───────────────

def parse_dates(data) -> Dict[str, str]:
    """
    Parsuje dates.json (schema: date.yml).
    use ∈ {initiation, announcement, publication, effect, notification,
            expiration, modification, other}

    - announcement → data ogłoszenia/wydania decyzji (date_issued)
    - publication  → data publikacji na portalu (date_published)
    - effect       → data wejścia w życie (date_effect)
    """
    result = {"date_issued": "", "date_published": "", "date_effect": ""}
    if not data:
        return result

    items = data if isinstance(data, list) else data.get("dates", [])
    for d in (items if isinstance(items, list) else []):
        if not isinstance(d, dict):
            continue
        use = d.get("use", "")
        val = d.get("date", "")
        if not val:
            continue
        if use == "announcement" and not result["date_issued"]:
            result["date_issued"] = val
        elif use == "publication" and not result["date_published"]:
            result["date_published"] = val
        elif use == "effect" and not result["date_effect"]:
            result["date_effect"] = val

    return result


# ─────────────────────────── PARSOWANIE refs.json ────────────────

# Mapowanie typów relacji na kategorie grafu
# Wg ref.yml: quotes, refers, implements, amends, changes, etc.
RELATION_TO_GRAPH = {
    "quotes":       "QUOTES",       # decyzja cytuje akt/orzeczenie
    "quoted":       "QUOTED_BY",
    "refers":       "REFERS",       # odwołuje się
    "referred":     "REFERRED_BY",
    "implements":   "IMPLEMENTS",   # implementuje rozporządzenie UE (RODO)
    "implemented":  "IMPLEMENTED_BY",
    "amends":       "AMENDS",
    "amended":      "AMENDED_BY",
    "executes":     "EXECUTES",
    "introduces":   "INTRODUCES",
    "replaces":     "REPLACES",
    "replaced":     "REPLACED_BY",
}

def parse_refs(data) -> Dict:
    """
    Parsuje refs z meta.json (pole refs[]).
    dates.json zwraca pusty body — nie używamy.

    Format (z surowego meta.json):
      refid    — "urn:ndoc:pro:eu:ojol:2016:679" / "urn:ndoc:pro:eu:ojol:2016:679r02" itp.
      name     — plain string: "I OSK 37/07" / "DKN.5131.10.2025" / "Dz.U. 2018 poz. 1000"
      relation — "refers" / "quotes" / "implements" / ...
      type     — "direct" / "indirect"
    """
    result = {
        "acts": [], "eu_acts": [], "court_rulings": [],
        "uodo_rulings": [], "edpb": [], "refs_full": [],
    }
    if not data:
        return result

    items = data if isinstance(data, list) else data.get("refs", [])

    for ref in (items if isinstance(items, list) else []):
        if not isinstance(ref, dict):
            continue

        refid    = ref.get("refid", "")
        relation = ref.get("relation", "refers")
        ref_type = ref.get("type", "direct")
        # name to plain string (nie obiekt {pl, en})
        name     = ref.get("name", "") or ""
        graph_rel = RELATION_TO_GRAPH.get(relation, "REFERS")

        entry = {
            "refid": refid, "relation": relation, "ref_type": ref_type,
            "graph_relation": graph_rel, "name": name,
        }

        if "urn:ndoc:pro:pl:durp:" in refid:
            # Polska ustawa — urn:ndoc:pro:pl:durp:2018:1000
            m = re.search(r"durp:(\d{4}):(\d+)", refid)
            if m:
                sig = f"Dz.U. {m.group(1)} poz. {m.group(2)}"
                entry["signature"] = sig
                entry["category"] = "act"
                if sig not in result["acts"]:
                    result["acts"].append(sig)
                result["refs_full"].append(entry)

        elif "urn:ndoc:pro:eu:ojol:" in refid:
            # Akt UE — urn:ndoc:pro:eu:ojol:2016:679 + warianty r02,r03 (sprostowania)
            # Wszystkie warianty tego samego aktu deduplikuj do EU RRRR/NR
            m = re.search(r"ojol:(\d{4}):(\d+)", refid)
            if m:
                sig = f"EU {m.group(1)}/{m.group(2)}"
                entry["signature"] = sig
                entry["category"] = "eu_act"
                if sig not in result["eu_acts"]:
                    result["eu_acts"].append(sig)
                    result["refs_full"].append(entry)
                # Warianty r02/r03 tego samego aktu: dodaj do refs_full ale nie duplikuj listy

        elif "urn:ndoc:court:" in refid:
            # Wyrok NSA/WSA — name to gotowa sygnatura "I OSK 37/07"
            sig = name if name else refid.split(":")[-1].replace("_", " ").upper()
            entry["signature"] = sig
            entry["category"] = "court_ruling"
            if sig not in result["court_rulings"]:
                result["court_rulings"].append(sig)
            result["refs_full"].append(entry)

        elif "urn:ndoc:gov:pl:uodo:" in refid:
            # Inna decyzja UODO — name to gotowa sygnatura "DKN.5131.10.2025"
            sig = name if name else refid_to_signature(refid)
            entry["signature"] = sig
            entry["category"] = "uodo_ruling"
            if sig not in result["uodo_rulings"]:
                result["uodo_rulings"].append(sig)
            result["refs_full"].append(entry)

        elif "urn:ndoc:gov:eu:edpb:" in refid:
            # Wytyczne EDPB — urn:ndoc:gov:eu:edpb:2022:04
            sig = name if name else refid.split(":")[-1]
            entry["signature"] = sig
            entry["category"] = "edpb"
            if sig not in result["edpb"]:
                result["edpb"].append(sig)
            result["refs_full"].append(entry)

        else:
            entry["signature"] = name or refid
            entry["category"] = "other"
            result["refs_full"].append(entry)

    return result


# ─────────────────────────── REFS Z TREŚCI (FALLBACK) ────────────

# Wzorce do wyciągania powiązań z content_text gdy refs.json jest pusty
_RE_DZ_U = re.compile(
    r'Dz\.\s*U\.\s*(?:z\s+)?(\d{4})\s+(?:r\.\s+)?poz\.\s+(\d+)'
)
_RE_RODO = re.compile(
    r'rozporządzeni[au]\s+(?:Parlamentu[^,]{0,60}?)?\(?(?:UE\s+)?2016/679'
)
_RE_UODO_SIG = re.compile(
    r'\b(DKN|ZSPU|ZSZS|ZKE)\.\d{4}\.\d+\.\d{4}\b'
)
_RE_DATE = re.compile(
    r'(?:z\s+dnia\s+|dnia\s+)(\d{1,2})\s+'
    r'(stycznia|lutego|marca|kwietnia|maja|czerwca|lipca|sierpnia|wrze\u015bnia|pa\u017adziernika|listopada|grudnia)'
    r'\s+(20[12]\d)\s+r\.'
)
_MONTHS = {
    'stycznia':1,'lutego':2,'marca':3,'kwietnia':4,'maja':5,'czerwca':6,
    'lipca':7,'sierpnia':8,'wrze\u015bnia':9,'pa\u017adziernika':10,'listopada':11,'grudnia':12
}

_RE_NSA = re.compile(
    r'\b(I|II|III|IV|V|VI|VII)\s+[A-Z]{2,4}/[A-Za-z]{2,4}\s+\d+/\d{4}\b'
)


def extract_date_from_text(content: str) -> str:
    """
    Wyciąga datę wydania decyzji z treści.
    Regex szuka dat z lat 2018+ (20[12]d) więc pomija daty historyczne
    z cytowań aktów (KPA 1960, KC 1964 itp.).
    Data decyzji jest w nagłówku — szukamy w pierwszych 500 znakach.
    """
    # Szukaj w samym nagłówku (przed "Na podstawie")
    header_end = content.find("Na podstawie")
    header = content[:header_end] if header_end > 0 else content[:500]
    m = _RE_DATE.search(header)
    if m:
        day, month_str, year = m.group(1), m.group(2), m.group(3)
        month = _MONTHS.get(month_str, 0)
        if month:
            return f"{year}-{month:02d}-{int(day):02d}"
    return ""


def extract_refs_from_text(content: str, doc_own_sig: str = "") -> Dict:
    """
    Wyciąga powiązania z treści decyzji gdy API refs.json jest puste.
    """
    result = {
        "acts": [], "eu_acts": [], "court_rulings": [], "uodo_rulings": [],
        "refs_full": [],
    }
    if not content:
        return result

    # Polskie akty: Dz. U. z 2025 r. poz. 1691
    for m in _RE_DZ_U.finditer(content):
        sig = f"Dz.U. {m.group(1)} poz. {m.group(2)}"
        if sig not in result["acts"]:
            result["acts"].append(sig)
            result["refs_full"].append({
                "refid": "", "signature": sig, "category": "act",
                "relation": "quotes", "graph_relation": "QUOTES",
                "ref_type": "direct", "name": "", "title": "",
            })

    # RODO / rozporządzenie 2016/679
    if _RE_RODO.search(content):
        sig = "EU 2016/679"
        if sig not in result["eu_acts"]:
            result["eu_acts"].append(sig)
            result["refs_full"].append({
                "refid": "", "signature": sig, "category": "eu_act",
                "relation": "implements", "graph_relation": "IMPLEMENTS",
                "ref_type": "direct", "name": "RODO", "title": "",
            })

    # Sygnatury innych decyzji UODO (pomijamy własną sygnaturę dokumentu)
    own_sig = doc_own_sig  # przekazana z zewnątrz
    for m in _RE_UODO_SIG.finditer(content):
        sig = m.group(0)
        if sig != own_sig and sig not in result["uodo_rulings"]:
            result["uodo_rulings"].append(sig)
            result["refs_full"].append({
                "refid": "", "signature": sig, "category": "uodo_ruling",
                "relation": "refers", "graph_relation": "REFERS",
                "ref_type": "direct", "name": "", "title": "",
            })

    # Wyroki NSA/WSA
    for m in _RE_NSA.finditer(content):
        sig = m.group(0)
        if sig not in result["court_rulings"]:
            result["court_rulings"].append(sig)
            result["refs_full"].append({
                "refid": "", "signature": sig, "category": "court_ruling",
                "relation": "refers", "graph_relation": "REFERS",
                "ref_type": "direct", "name": "", "title": "",
            })

    return result


# ─────────────────────────── STATUS PRAWNY ───────────────────────

# Mapowanie pub_workflow_status → status prawny
# Wartości z serwera: "nonfinal", "final" (schemat mówił published/archived/draft —
# serwer używa innych wartości)
_PUB_STATUS_MAP = {
    "final":    "prawomocna",
    "nonfinal": "nieprawomocna",
    "published": "prawomocna",   # fallback ze schematu
    "archived":  "prawomocna",
}


def extract_legal_status(keywords: str, pub_status: str) -> str:
    """
    Status prawny decyzji (prawomocna/nieprawomocna).
    Priorytet: pub_workflow_status > keywords.
    """
    if pub_status in _PUB_STATUS_MAP:
        return _PUB_STATUS_MAP[pub_status]
    kw_lower = keywords.lower()
    if "prawomocna" in kw_lower:
        return "prawomocna"
    return "nieprawomocna"


# ─────────────────────────── POBIERANIE DOKUMENTU ────────────────

def fetch_decision(session: requests.Session,
                   doc_id: str,
                   doc_fields: Dict,
                   delay: float = DEFAULT_DELAY) -> Dict:
    """
    Pobiera pełne dane orzeczenia.

    doc_fields — dane z search: {id, refid, keywords, title_pl,
                                  date_announcement, date_publication}
    """
    refid = doc_fields.get("refid", "")
    if not refid:
        return {"_error": "brak_refid", "doc_id": doc_id}

    sig = refid_to_signature(refid)

    doc = {
        "doc_id": doc_id,
        "refid": refid,
        "signature": sig,
        "url": f"https://orzeczenia.uodo.gov.pl/document/{refid}/content",
        "source_collection": "UODO",
        "title": "",          # krótka nazwa z name.pl
        "title_full": "",     # pełny opis naruszenia z title.pl
        "keywords": "",       # string z przecinkami (do embeddingu)
        "keywords_list": [],  # lista (do filtrowania w Qdrant)
        "status": "",         # prawomocna / nieprawomocna
        "pub_workflow_status": "",  # published / archived / draft...
        "kind": "",           # decision / judgment / ...
        "date_issued": "",
        "date_published": "",
        "date_effect": "",
        "year": 0,
        "entities": [],       # podmioty: [{title, name, function}]
        "content_text": "",
        "meta": {},
        "refs_from_content": {
            "acts": [], "eu_acts": [], "court_rulings": [], "uodo_rulings": []
        },
        "refs_full": [],      # pełna lista powiązań z typem relacji
        "related_legislation": [],
        "related_rulings": [],
        # Taksonomia UODO — rozbita wg grup labelów
        "term_decision_type": [],
        "term_violation_type": [],
        "term_legal_basis": [],
        "term_corrective_measure": [],
        "term_sector": [],
    }

    # Rok z sygnatury
    year_m = re.search(r"\b(20\d{2})\b", sig)
    doc["year"] = int(year_m.group(1)) if year_m else 0

    # ── Dane z search ──────────────────────────────────────────
    kw_raw = doc_fields.get("keywords", "")
    doc["keywords"] = ", ".join(kw_raw) if isinstance(kw_raw, list) else str(kw_raw or "")
    doc["title"] = doc_fields.get("title_pl", "")
    # date_announcement z search = None w praktyce — pobierzemy z dates.json

    # ── 1. Treść (body.txt) ────────────────────────────────────
    # refpath format: {refid}:{part} — part 0 = treść główna
    r = get(session, f"{API_BASE}/documents/public/items/{refid}:0/body.txt",
            accept="text/plain")
    time.sleep(delay)
    if r:
        doc["content_text"] = r.text
        print(f"  ✅ body: {len(doc['content_text'])} chars")
    else:
        # Fallback bez numeru części
        r = get(session, f"{API_BASE}/documents/public/items/{refid}/body.txt",
                accept="text/plain")
        time.sleep(delay)
        if r:
            doc["content_text"] = r.text
            print(f"  ✅ body (fallback): {len(doc['content_text'])} chars")
        else:
            print("  ⚠️ brak treści")

    # ── 2. Metadane (meta.json) ────────────────────────────────
    # Zawiera: name, title (pełny), terms (keywords), entities, kind
    r = get(session, f"{API_BASE}/documents/public/items/{refid}/meta.json")
    time.sleep(delay)
    if r:
        meta_raw = r.json()
        doc["meta"] = meta_raw
        parsed = parse_meta(meta_raw)

        # name.pl → krótka nazwa (jeśli search nie dał title_pl)
        if not doc["title"] and parsed["name"]:
            doc["title"] = parsed["name"]

        # title.pl → pełny opis naruszenia
        doc["title_full"] = parsed["title_full"] or doc["title"]

        # keywords z terms[] (bogatsze niż z indeksu search)
        if parsed["keywords"]:
            doc["keywords"] = parsed["keywords"]
            doc["keywords_list"] = parsed["keywords_list"]
        else:
            # Fallback — sparsuj string z search
            doc["keywords_list"] = [
                k.strip() for k in doc["keywords"].split(",") if k.strip()
            ]

        doc["entities"] = parsed["entities"]
        doc["kind"] = parsed["kind"]
        doc["pub_workflow_status"] = parsed["pub_workflow_status"]
        # Daty z meta.json (date_announcement, date_publication)
        if parsed["date_issued"]:
            doc["date_issued"] = parsed["date_issued"]
        if parsed["date_published"]:
            doc["date_published"] = parsed["date_published"]
        # Status prawny — bezpośrednie pole z meta, fallback z pub_workflow_status
        if parsed["legal_status"]:
            doc["status"] = parsed["legal_status"]
        else:
            doc["status"] = extract_legal_status(doc["keywords"], doc["pub_workflow_status"])
        # Taksonomia UODO
        doc["term_decision_type"]      = parsed["term_decision_type"]
        doc["term_violation_type"]     = parsed["term_violation_type"]
        doc["term_legal_basis"]        = parsed["term_legal_basis"]
        doc["term_corrective_measure"] = parsed["term_corrective_measure"]
        doc["term_sector"]             = parsed["term_sector"]

        kw_count = len(doc["keywords_list"])
        print(f"  ✅ meta: keywords={kw_count}, kind={doc['kind']}, status={doc['status']}, issued={doc['date_issued']}")

    # ── 3. Daty (dates.json, fallback: z treści) ─────────────
    r = get(session, f"{API_BASE}/documents/public/items/{refid}/dates.json")
    time.sleep(delay)
    if r:
        dates = parse_dates(r.json())
        doc["date_issued"] = dates["date_issued"]
        doc["date_published"] = dates["date_published"]
        doc["date_effect"] = dates["date_effect"]
    # Fallback: data z treści decyzji (format "z dnia DD miesiąca YYYY r.")
    if not doc["date_issued"] and doc.get("content_text"):
        doc["date_issued"] = extract_date_from_text(doc["content_text"])
    if doc["date_issued"]:
        doc["year"] = int(doc["date_issued"][:4])
        print(f"  ✅ dates: issued={doc['date_issued']}")
    else:
        print(f"  ⚠️ brak daty")

    # ── 4. Powiązania z meta.json (refs.json endpoint jest pusty) ──
    refs = parsed.get("refs", {}) if parsed else {}
    if refs.get("acts") or refs.get("eu_acts") or refs.get("uodo_rulings") or refs.get("court_rulings"):
        doc["refs_from_content"] = {
            "acts": refs["acts"], "eu_acts": refs["eu_acts"],
            "court_rulings": refs["court_rulings"], "uodo_rulings": refs["uodo_rulings"],
        }
        doc["refs_full"] = refs["refs_full"]
        doc["related_legislation"] = [
            {"type": "act",    "signature": s, "relation": _find_relation(refs["refs_full"], s)}
            for s in refs["acts"]
        ] + [
            {"type": "eu_act", "signature": s, "relation": _find_relation(refs["refs_full"], s)}
            for s in refs["eu_acts"]
        ]
        doc["related_rulings"] = [
            {"type": "uodo_ruling",  "signature": s, "relation": _find_relation(refs["refs_full"], s)}
            for s in refs["uodo_rulings"]
        ] + [
            {"type": "court_ruling", "signature": s, "relation": _find_relation(refs["refs_full"], s)}
            for s in refs["court_rulings"]
        ]
        print(
            f"  ✅ refs (meta): acts={len(refs['acts'])}, eu={len(refs['eu_acts'])}, "
            f"uodo={len(refs['uodo_rulings'])}, courts={len(refs['court_rulings'])}, "
            f"edpb={len(refs.get('edpb', []))}"
        )

    # ── Fallback: wyciągnij powiązania z treści jeśli meta nic nie dało ──
    total_refs = (len(doc["refs_from_content"]["acts"]) +
                  len(doc["refs_from_content"]["eu_acts"]) +
                  len(doc["refs_from_content"]["uodo_rulings"]) +
                  len(doc["refs_from_content"]["court_rulings"]))
    if total_refs == 0 and doc.get("content_text"):
        refs = extract_refs_from_text(doc["content_text"], doc_own_sig=sig)
        doc["refs_from_content"] = {
            "acts": refs["acts"], "eu_acts": refs["eu_acts"],
            "court_rulings": refs["court_rulings"], "uodo_rulings": refs["uodo_rulings"],
        }
        doc["refs_full"] = refs["refs_full"]
        doc["related_legislation"] = [
            {"type": "act",    "signature": s, "relation": "quotes"} for s in refs["acts"]
        ] + [
            {"type": "eu_act", "signature": s, "relation": "implements"} for s in refs["eu_acts"]
        ]
        doc["related_rulings"] = [
            {"type": "uodo_ruling",  "signature": s, "relation": "refers"} for s in refs["uodo_rulings"]
        ] + [
            {"type": "court_ruling", "signature": s, "relation": "refers"} for s in refs["court_rulings"]
        ]
        print(
            f"  ✅ refs (z treści): acts={len(refs['acts'])}, eu={len(refs['eu_acts'])}, "
            f"uodo={len(refs['uodo_rulings'])}, courts={len(refs['court_rulings'])}"
        )

    return doc


def _find_relation(refs_full: List[Dict], signature: str) -> str:
    """Znajdź typ relacji dla danej sygnatury w refs_full."""
    for r in refs_full:
        if r.get("signature") == signature:
            return r.get("relation", "refers")
    return "refers"


# ─────────────────────────── GŁÓWNA PĘTLA ────────────────────────

def scrape_all(output_path: str,
               user: str = None, password: str = None,
               delay: float = DEFAULT_DELAY,
               resume: bool = True,
               date_from: str = None, date_to: str = None,
               limit: int = None):

    session = make_session(user, password)

    done = set()
    if resume and os.path.exists(output_path):
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    done.add(d.get("doc_id", ""))
                    done.add(d.get("refid", ""))
                except Exception:
                    pass
        print(f"🔄 Resume: {len(done) // 2} już pobranych")

    all_docs = fetch_document_list(session, date_from, date_to)
    if limit:
        all_docs = all_docs[:limit]

    to_scrape = [d for d in all_docs if d.get("id", "") not in done]
    print(f"📝 Do pobrania: {len(to_scrape)}")

    if not to_scrape:
        print("✅ Wszystko gotowe!")
        return

    errors = 0
    with open(output_path, "a", encoding="utf-8") as out_f:
        for i, doc_fields in enumerate(to_scrape, 1):
            doc_id = doc_fields.get("id", "")
            refid = doc_fields.get("refid", "")
            sig = refid_to_signature(refid) if refid else doc_id
            print(f"\n[{i}/{len(to_scrape)}] 🔍 {sig}")

            doc = fetch_decision(session, doc_id, doc_fields, delay=delay)

            if doc.get("_error"):
                print(f"  ❌ {doc['_error']}")
                errors += 1
                continue

            out_f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            out_f.flush()

            if i % 50 == 0:
                print(f"\n📊 {i}/{len(to_scrape)} ({i/len(to_scrape)*100:.1f}%), błędy: {errors}")

    print(f"\n✅ Gotowe! błędy: {errors}/{len(to_scrape)}")


# ─────────────────────────── CLI ─────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UODO API Scraper")
    parser.add_argument("--output", default="uodo_decisions.jsonl")
    parser.add_argument("--user",     default=None)
    parser.add_argument("--password", default=None)
    parser.add_argument("--delay",    type=float, default=DEFAULT_DELAY)
    parser.add_argument("--no-resume",   action="store_true")
    parser.add_argument("--date-from",   default=None, metavar="YYYY-MM-DD")
    parser.add_argument("--date-to",     default=None, metavar="YYYY-MM-DD")
    parser.add_argument("--test",        action="store_true",
                        help="Pobierz 3 dokumenty testowe")
    args = parser.parse_args()

    if args.test:
        out = "uodo_test.jsonl"
        if os.path.exists(out):
            os.remove(out)
        scrape_all(out, args.user, args.password,
                   delay=args.delay, resume=False, limit=3)

        print("\n=== WYNIKI TESTU ===")
        with open(out) as f:
            for line in f:
                d = json.loads(line)
                refs = d.get("refs_from_content", {})
                print(f"\n{'='*60}")
                print(f"Sygnatura:    {d['signature']}")
                print(f"Kind:         {d.get('kind', '')}")
                print(f"Tytuł:        {d.get('title', '')[:100]}")
                print(f"Tytuł pełny:  {d.get('title_full', '')[:120]}")
                print(f"Data wydania: {d.get('date_issued', '')}")
                print(f"Status:       {d.get('status', '')}")
                print(f"Pub status:   {d.get('pub_workflow_status', '')}")
                print(f"Treść:        {len(d.get('content_text', ''))} chars")
                print(f"Preview:      {d.get('content_text', '')[:200]}")
                print(f"Keywords:     {len(d.get('keywords_list', []))} — {d.get('keywords_list', [])[:5]}")
                print(f"Entities:     {d.get('entities', [])}")
                print(f"Acts:         {refs.get('acts', [])[:3]}")
                print(f"EU acts:      {refs.get('eu_acts', [])[:3]}")
                print(f"UODO refs:    {refs.get('uodo_rulings', [])[:3]}")
                print(f"Courts:       {refs.get('court_rulings', [])[:3]}")
                print(f"Refs full:    {len(d.get('refs_full', []))} powiązań")
                if d.get("refs_full"):
                    for r in d["refs_full"][:3]:
                        print(f"  {r.get('relation'):12} {r.get('category'):12} {r.get('signature','')}")
    else:
        scrape_all(
            args.output, args.user, args.password,
            delay=args.delay,
            resume=not args.no_resume,
            date_from=args.date_from,
            date_to=args.date_to,
        )
