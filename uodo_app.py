#!/usr/bin/env python3
"""
UODO RAG Demo — wyszukiwarka decyzji Prezesa UODO + ustawa o ochronie danych osobowych.

Uruchomienie:
  streamlit run uodo_app.py

Wymagania:
  pip install streamlit qdrant-client sentence-transformers networkx groq requests python-dotenv
"""

import os
import pickle
import re
import time
from collections.abc import Generator
from typing import Any

import networkx as nx
import streamlit as st
from jinja2 import Environment
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue, Range

# Regex dopasowujący sygnaturę decyzji UODO wpisaną bezpośrednio jako query
_RE_QUERY_SIG = re.compile(r"^\s*([A-Z]{2,6}\.\d{3,5}\.\d+\.\d{4})\s*$", re.IGNORECASE)

# ─────────────────────────── CONFIG ──────────────────────────────

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "uodo_decisions"
GRAPH_PATH = os.getenv("UODO_GRAPH_PATH", "./uodo_graph.pkl")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sdadas/mmlw-retrieval-roberta-large")

# Wczytaj .env jeśli istnieje
try:
    from dotenv import load_dotenv

    _ = load_dotenv()
except ImportError:
    pass

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OLLAMA_CLOUD_API_KEY = os.getenv("OLLAMA_CLOUD_API_KEY", "")
OLLAMA_CLOUD_URL = os.getenv("OLLAMA_CLOUD_URL", "https://ollama.com")
OLLAMA_LOCAL_URL = os.getenv("OLLAMA_LOCAL_URL", "http://localhost:11434")

PROVIDERS = ["Ollama Cloud", "Groq"]
DEFAULT_PROVIDER = "Ollama Cloud"
DEFAULT_OLLAMA_MODEL = "kimi-k2.5"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"

TOP_K = 8
GRAPH_DEPTH = 2
UODO_PORTAL_BASE = "https://orzeczenia.uodo.gov.pl/document"
ISAP_ACT_URL = "https://isap.sejm.gov.pl/isap.nsf/DocDetails.xsp?id=WDU20190001781"
GDPR_URL = "https://eur-lex.europa.eu/legal-content/PL/TXT/?uri=CELEX:32016R0679"


# ─────────────────────────── MODELE STANU ────────────────────────
# Wzorzec z kursu Software 3.0: Reasoning Step (moduł 4) + Memory (moduł 2)

from enum import Enum

from pydantic import BaseModel, Field


class QueryType(str, Enum):
    DECISION_LOOKUP = "szukam_decyzji"  # zapytanie o konkretną decyzję/karę
    LEGAL_ARTICLE = "szukam_przepisu"  # pytanie o artykuł ustawy/RODO
    GENERAL_ANALYSIS = "analiza_ogólna"  # szeroka analiza tematu
    FACTUAL = "pytanie_faktyczne"  # kto/kiedy/ile


class QueryDecomposition(BaseModel):
    """Reasoning Step — LLM dekompozycja pytania PRZED wyszukiwaniem.
    Wzorzec z lekcji 4.1: zamiast szukać surowej frazy, model najpierw
    analizuje intencję i generuje ustrukturyzowane parametry wyszukiwania.
    """

    original_query: str
    query_type: QueryType = QueryType.GENERAL_ANALYSIS
    search_keywords: list[str] = Field(
        default_factory=list,
        description="Synonimy i pojęcia prawne do wyszukiwania (max 5)",
    )
    gdpr_articles_hint: list[str] = Field(
        default_factory=list,
        description="Artykuły RODO które mogą być istotne (np. ['Art. 5', 'Art. 83'])",
    )
    uodo_act_articles_hint: list[str] = Field(
        default_factory=list, description="Artykuły u.o.d.o. które mogą być istotne"
    )
    year_from_hint: int | None = None
    year_to_hint: int | None = None
    enriched_query: str = Field(
        description="Rozszerzone zapytanie do wyszukiwania semantycznego"
    )
    reasoning: str = Field(
        description="Krótkie uzasadnienie dekompozycji (widoczne w UI)"
    )


class MemoryEntry(BaseModel):
    """Wpis w pamięci epizodycznej."""

    query: str
    enriched_query: str
    decomposition_summary: str
    top_signatures: list[str] = []  # sygnatury top decyzji
    top_articles: list[str] = []  # numery artykułów
    answer_snippet: str = ""  # pierwsze 300 znaków odpowiedzi AI


class AgentMemory(BaseModel):
    """Pamięć epizodyczna sesji — wzorzec z lekcji 2.1 Memory Engineering.
    Przechowuje ostatnie N analiz; używana do:
    - wzbogacania kontekstu o poprzednie wyniki (bez ponownego wyszukiwania)
    - pokazania użytkownikowi historii sesji
    """

    entries: list[MemoryEntry] = []
    max_entries: int = 5

    def add(self, entry: MemoryEntry) -> None:
        self.entries.insert(0, entry)
        self.entries = self.entries[: self.max_entries]

    def find_related(self, query: str) -> list[MemoryEntry]:
        """Prosta heurystyka: wpisy z co najmniej jednym wspólnym słowem kluczowym."""
        q_words = {w.lower() for w in re.split(r"\W+", query) if len(w) > 3}
        result = []
        for e in self.entries:
            e_words = {w.lower() for w in re.split(r"\W+", e.query) if len(w) > 3}
            if q_words & e_words:
                result.append(e)
        return result


# ─────────────────────────── JINJA2 SZABLONY KONTEKSTU ────────────
# Wzorzec z lekcji 1.2: Context Engineering z szablonami Jinja2
# Zamiast sklejać f-stringi, każdy typ dokumentu ma własny szablon.
# Pozwala na precyzyjną kontrolę "zakotwiczenia uwagi" modelu.

_JINJA_ENV = Environment(keep_trailing_newline=True)

# POPRAWKA 1: Nagłówek jawnie wymienia WSZYSTKIE typy dokumentów obecne w kontekście
# i priorytyzuje decyzje UODO — duże modele (kimi2.5, gpt-4o) czytają nagłówek
# dosłownie i jeśli nie widzą w nim RODO, traktują artykuły RODO jako "nie-decyzje".
_TPL_HEADER = _JINJA_ENV.from_string("""Poniżej znajdują się dokumenty powiązane z pytaniem: «{{ query }}»
Zbiór zawiera trzy typy dokumentów:
  1. DECYZJE UODO — decyzje administracyjne Prezesa Urzędu Ochrony Danych Osobowych
  2. ARTYKUŁY u.o.d.o. — przepisy ustawy o ochronie danych osobowych (Dz.U. 2019 poz. 1781)
  3. ARTYKUŁY RODO — przepisy rozporządzenia (UE) 2016/679 (Dz.Urz. UE L 119/1)
Każdy dokument jest wyraźnie oznaczony typem w nagłówku bloku.
{% if filter_note %}{{ filter_note }}{% endif %}
{% if memory_note %}{{ memory_note }}{% endif %}
Odpowiadaj na podstawie poniższych dokumentów, ze szczególnym uwzględnieniem DECYZJI UODO.
Podawaj sygnatury decyzji [np. DKN.XXX.X.XXXX, ZSOŚS, i in.] i numery artykułów [np. Art. X u.o.d.o.].
""")

_TPL_DECISION = _JINJA_ENV.from_string("""[{{ rank }}] DECYZJA UODO {{ sig }} ({{ date }}, {{ status }}){% if graph_rel %} [powiązana: {{ graph_rel }}]{% endif %}

  SYGNATURA:    {{ sig }}
  DATA:         {{ date }}
  STATUS:       {{ status }}
{% if keywords %}  TAGI:         {{ keywords }}
{% endif %}{% if acts %}  POWOŁANE AKTY: {{ acts }}
{% endif %}  TREŚĆ:
{{ fragment }}
""")

_TPL_ACT_ARTICLE = _JINJA_ENV.from_string("""[{{ rank }}] USTAWA o ochronie danych osobowych — Art. {{ art_num }}{% if label_suffix %} {{ label_suffix }}{% endif %}

  ŹRÓDŁO: Dz.U. 2019 poz. 1781 (u.o.d.o.)
  TREŚĆ:
{{ text }}
""")

_TPL_GDPR = _JINJA_ENV.from_string("""[{{ rank }}] RODO (rozporządzenie 2016/679) — {{ prefix }}

  ŹRÓDŁO: Dz.Urz. UE L 119/1
  TREŚĆ:
{{ text }}
""")


# ─────────────────────────── CACHE / ZASOBY ──────────────────────


@st.cache_resource
def get_qdrant() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, timeout=30)


@st.cache_resource
def get_embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBED_MODEL, trust_remote_code=True)


@st.cache_resource
def get_graph() -> nx.DiGraph | None:  # type: ignore[type-arg]
    if os.path.exists(GRAPH_PATH):
        with open(GRAPH_PATH, "rb") as f:
            return pickle.load(f)

    G = nx.DiGraph()
    client = get_qdrant()
    offset = None
    while True:
        pts, next_off = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=500,
            offset=offset,
            with_payload=[
                "signature",
                "doc_type",
                "related_uodo_rulings",
                "related_acts",
                "related_eu_acts",
                "related_court_rulings",
            ],
            with_vectors=False,
        )
        for p in pts:
            pay = p.payload or {}
            sig = pay.get("signature", "")
            dtype = pay.get("doc_type", "")
            if not sig or dtype != "uodo_decision":
                continue
            G.add_node(sig, doc_type="uodo_decision", qdrant_id=str(p.id))
            for rel_sig in pay.get("related_uodo_rulings", []):
                if not G.has_node(rel_sig):
                    G.add_node(rel_sig, doc_type="uodo_decision")
                G.add_edge(sig, rel_sig, relation="CITES_UODO")
            for rel_sig in pay.get("related_acts", []):
                if not G.has_node(rel_sig):
                    G.add_node(rel_sig, doc_type="act")
                G.add_edge(sig, rel_sig, relation="CITES_ACT")
            for rel_sig in pay.get("related_eu_acts", []):
                if not G.has_node(rel_sig):
                    G.add_node(rel_sig, doc_type="eu_act")
                G.add_edge(sig, rel_sig, relation="CITES_EU")
        if next_off is None:
            break
        offset = next_off

    with open(GRAPH_PATH, "wb") as f:
        pickle.dump(G, f)
    return G


# ─────────────────────────── WYSZUKIWANIE ────────────────────────


def embed(text: str) -> list[float]:
    return get_embedder().encode(text, normalize_embeddings=True).tolist()


def semantic_search(
    query: str,
    top_k: int = TOP_K,
    filters: dict[str, Any] | None = None,
    score_threshold: float = 0.25,
) -> list[dict[str, Any]]:
    vec = embed(query)
    client = get_qdrant()

    must = []
    if filters:
        if filters.get("status"):
            must.append(
                FieldCondition(key="status", match=MatchValue(value=filters["status"]))
            )

        if filters.get("keyword"):
            must.append(
                FieldCondition(
                    key="keywords", match=MatchValue(value=filters["keyword"])
                )
            )
        if filters.get("doc_types"):
            must.append(
                FieldCondition(key="doc_type", match=MatchAny(any=filters["doc_types"]))
            )
        for term_field in (
            "term_decision_type",
            "term_violation_type",
            "term_legal_basis",
            "term_corrective_measure",
            "term_sector",
        ):
            vals = filters.get(term_field, [])
            if vals:
                must.append(FieldCondition(key=term_field, match=MatchAny(any=vals)))

    qdrant_filter = Filter(must=must) if must else None

    res = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vec,
        limit=top_k,
        query_filter=qdrant_filter,
        with_payload=True,
        score_threshold=score_threshold,
    )

    docs = []
    for hit in res.points or []:
        d = (hit.payload or {}).copy()
        d["_score"] = hit.score
        d["_source"] = "semantic"
        docs.append(d)
    return docs


def graph_expand(
    seed_sigs: list[str], depth: int = GRAPH_DEPTH
) -> list[tuple[str, str, float]]:
    G = get_graph()
    if G is None:
        return []

    visited = set(seed_sigs)
    result = []
    frontier = set(seed_sigs)

    for d in range(depth):
        decay = 0.65**d
        new_frontier = set()
        for node in frontier:
            if node not in G:
                continue
            for nb in G.successors(node):
                if nb in visited:
                    continue
                if G[node][nb].get("relation") == "CITES_UODO":
                    result.append((nb, "cytowana", 0.6 * decay))
                    visited.add(nb)
                    new_frontier.add(nb)
            for nb in G.predecessors(node):
                if nb in visited:
                    continue
                if (
                    G[nb][node].get("relation") == "CITES_UODO"
                    and G.nodes.get(nb, {}).get("doc_type") == "uodo_decision"
                ):
                    result.append((nb, "cytuje tę decyzję", 0.5 * decay))
                    visited.add(nb)
                    new_frontier.add(nb)
        frontier = new_frontier
        if not frontier or len(result) >= 20:
            break

    result.sort(key=lambda x: -x[2])
    return result[:15]


def fetch_by_signature(sig: str) -> dict[str, Any] | None:
    client = get_qdrant()
    pts, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(
            must=[
                FieldCondition(key="signature", match=MatchValue(value=sig)),
                FieldCondition(key="doc_type", match=MatchValue(value="uodo_decision")),
            ]
        ),
        limit=1,
        with_payload=True,
    )
    if pts:
        d = (pts[0].payload or {}).copy()
        d["_source"] = "graph"
        d["_score"] = 0.0
        return d
    return None


def keyword_exact_search(
    keyword: str, filters: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Pobiera WSZYSTKIE dokumenty z danym tagiem (scroll z paginacją)."""
    client = get_qdrant()
    must = [FieldCondition(key="keywords", match=MatchValue(value=keyword))]
    if filters:
        if filters.get("status"):
            must.append(
                FieldCondition(key="status", match=MatchValue(value=filters["status"]))
            )
        if filters.get("year_from") or filters.get("year_to"):
            must.append(
                FieldCondition(
                    key="year",
                    range=Range(
                        gte=filters.get("year_from", 2000),
                        lte=filters.get("year_to", 2030),
                    ),
                )
            )
        if filters.get("doc_types"):
            must.append(
                FieldCondition(key="doc_type", match=MatchAny(any=filters["doc_types"]))
            )

    qdrant_filter = Filter(must=must) if must else None
    docs = []
    offset = None
    while True:
        pts, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=qdrant_filter,
            limit=100,
            offset=offset,
            with_payload=True,
        )
        for pt in pts or []:
            d = (pt.payload or {}).copy()
            d["_score"] = 1.0
            d["_source"] = "keyword"
            docs.append(d)
        if next_offset is None or not pts:
            break
        offset = next_offset
    return docs


# ── Taksonomia portalu UODO — wartości z plików txKind.json, txSector.json,
#    txRemedyRODO.json, taxo-infringement.json, taxo-base.json ──────────────
_TAXONOMY_STATIC: dict[str, list[str]] = {
    "term_decision_type": [
        "nakaz",
        "odmowa",
        "umorzenie",
        "upomnienie",
        "inne",
    ],
    "term_sector": [
        "BIP",
        "DODO",
        "Finanse",
        "Marketing",
        "Mieszkalnictwo",
        "Monitoring",
        "Pozostałe",
        "Szkolnictwo",
        "Telekomunikacja",
        "Ubezpieczenia",
        "Zatrudnienie",
        "Zdrowie",
    ],
    "term_corrective_measure": [
        "ostrzeżenie",
        "upomnienie",
        "nakaz spełnienia żądania",
        "dostosowanie",
        "poinformowanie",
        "ograniczenie przetwarzania",
        "sprostowanie/usunięcie/ograniczenie",
        "cofnięcie certyfikacji",
        "administracyjna kara pieniężna",
        "państwo trzecie",
    ],
    "term_violation_type": [],  # wypełniane dynamicznie z Qdrant (zbyt duże)
    "term_legal_basis": [],  # j.w.
}


@st.cache_data(ttl=3600, show_spinner=False)
def _get_taxonomy_options() -> dict[str, list[str]]:
    """Zwraca opcje filtrów taksonomii.
    Dla term_decision_type / term_sector / term_corrective_measure używa stałych
    wartości z taksonomii portalu UODO. Dla term_violation_type i term_legal_basis
    pobiera unikalne wartości z Qdrant (zbyt wiele wariantów do hardkodowania).
    """
    result = {k: list(v) for k, v in _TAXONOMY_STATIC.items()}
    dynamic_fields = [f for f, v in _TAXONOMY_STATIC.items() if not v]
    if not dynamic_fields:
        return result
    try:
        client = get_qdrant()
        offset = None
        while True:
            pts, next_off = client.scroll(
                collection_name=COLLECTION_NAME,
                limit=500,
                offset=offset,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="doc_type", match=MatchValue(value="uodo_decision")
                        )
                    ]
                ),
                with_payload=dynamic_fields,
                with_vectors=False,
            )
            for pt in pts or []:
                pay = pt.payload or {}
                for field in dynamic_fields:
                    for val in pay.get(field) or []:
                        if val and val not in result[field]:
                            result[field].append(val)
            if not next_off or not pts:
                break
            offset = next_off
        for field in dynamic_fields:
            result[field] = sorted(result[field])
    except Exception:
        pass
    return result


@st.cache_data(ttl=300, show_spinner=False)
def _get_all_tags() -> list[str]:
    """Pobiera wszystkie unikalne tagi z kolekcji (cache 5 min)."""
    client = get_qdrant()
    all_tags = set()
    offset = None
    while True:
        pts, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=500,
            with_payload=["keywords"],
            with_vectors=False,
            offset=offset,
        )
        for pt in pts or []:
            kws = (pt.payload or {}).get("keywords", [])
            if isinstance(kws, list):
                all_tags.update(kws)
            elif isinstance(kws, str):
                all_tags.update(k.strip() for k in kws.split(",") if k.strip())
        if not next_offset or not pts:
            break
        offset = next_offset
    return sorted(all_tags)


def _extract_tags_with_llm(query: str, available_tags: list[str]) -> list[str]:
    """Pyta LLM o tagi pasujące do zapytania.
    Zwraca maks. 8 tagów z listy + maks. 2 nowe tagi spoza listy (oznaczone [NOWY])."""
    provider = st.session_state.get("llm_provider", DEFAULT_PROVIDER)
    api_key = st.session_state.get("llm_api_key", "")
    model = st.session_state.get("llm_model", "")

    tags_list = "\n".join(f"- {t}" for t in available_tags)
    prompt = (
        f"Masz listę tagów z bazy orzeczeń UODO (organ ochrony danych osobowych w Polsce).\n"
        f"Wybierz tagi NAJBARDZIEJ pasujące do zapytania — maksymalnie 8 tagów z listy.\n"
        f"Jeśli temat zapytania nie jest pokryty przez żaden istniejący tag, możesz dodać maksymalnie 4 NOWE tagi spoza listy.\n"
        f"Uwzględnij synonimy i formy fleksyjne (np. 'kampania wyborcza' → szukaj tagów o wyborach, partiach, polityce).\n"
        f"Wybieraj tylko tagi ŚCIŚLE związane z tematem — nie wybieraj zbyt ogólnych tagów, chyba że zapytanie wprost o nie pyta.\n"
        f"Odpowiedz TYLKO listą tagów, jeden na linię, bez komentarzy.\n"
        f"Tagi z listy — dokładna pisownia. Nowe tagi — z prefiksem [NOWY].\n"
        f"Zapytanie: {query}\n\n"
        f"Dostępne tagi:\n{tags_list}"
    )

    try:
        if provider == "Groq":
            from groq import Groq

            client = Groq(api_key=api_key or GROQ_API_KEY)
            resp = client.chat.completions.create(
                model=model,
                max_tokens=400,
                stream=False,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.choices[0].message.content or ""
        else:
            import requests as _req

            resp = _req.post(
                f"{OLLAMA_CLOUD_URL}/api/chat",
                headers={"Authorization": f"Bearer {api_key or OLLAMA_CLOUD_API_KEY}"},
                json={
                    "model": model,
                    "stream": False,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=30,
            )
            raw = resp.json().get("message", {}).get("content", "")

        tags_lower = {t.lower(): t for t in available_tags}
        existing_found = []
        new_found = []

        for line in raw.strip().splitlines():
            line = line.strip().lstrip("- ").strip()
            if not line:
                continue
            if line.startswith("[NOWY]"):
                tag = line[6:].strip()
                if tag and len(tag) > 2 and len(new_found) < 2:
                    new_found.append(tag)
            elif line.lower() in tags_lower and len(existing_found) < 8:
                existing_found.append(tags_lower[line.lower()])

        return existing_found + new_found
    except Exception:
        return []


def _get_matched_tags(query: str) -> list[str]:
    """Zwraca listę tagów pasujących do zapytania (przez LLM)."""
    available_tags = _get_all_tags()
    return _extract_tags_with_llm(query, available_tags)


def _doc_key(d: dict[str, Any]) -> str:
    """Unikalny klucz dokumentu do deduplikacji."""
    # Dla artykułów ustawy i RODO użyj doc_id lub kombinacji pól
    doc_id = d.get("doc_id", "")
    if doc_id:
        return doc_id
    sig = d.get("signature", "")
    dtype = d.get("doc_type", "")
    art = d.get("article_num", "")
    chunk = d.get("chunk_index", 0)
    if dtype in ("legal_act_article", "gdpr_article", "gdpr_recital"):
        return f"{dtype}:{sig}:{art}:{chunk}"
    return sig or f"{dtype}:{art}"


def hybrid_search(
    query: str,
    top_k: int = TOP_K,
    filters: dict[str, Any] | None = None,
    use_graph: bool = True,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Zwraca (list[dict], list[str]) — dokumenty i użyte tagi.

    Priorytety:
      1. Decyzje UODO — zwracamy WSZYSTKIE pasujące (bez limitu top_k)
      2. Artykuły u.o.d.o. — max MAX_ACT_DOCS
      3. Artykuły RODO — max MAX_GDPR_DOCS
    """
    MAX_ACT_DOCS = 5
    MAX_GDPR_DOCS = 3

    matched_tags = _get_matched_tags(query)

    seen_keys: set[str] = set()
    decisions: list[dict[str, Any]] = []
    act_docs: list[dict[str, Any]] = []
    gdpr_docs: list[dict[str, Any]] = []

    def _add(bucket: list, doc: dict) -> bool:
        key = _doc_key(doc)
        if key in seen_keys:
            return False
        seen_keys.add(key)
        bucket.append(doc)
        return True

    # ── Filtry bazowe bez pola keyword (keyword stosujemy osobno) ─────────
    filters_base = {k: v for k, v in (filters or {}).items() if k != "keyword"}

    # ═══════════════════════════════════════════════════════════════════════
    # DECYZJE UODO — pełny recall (keyword_exact_search = scroll bez limitu)
    # ═══════════════════════════════════════════════════════════════════════

    # ── 1a. Explicit keyword z UI → scroll po tagu, tylko decyzje ────────
    explicit_keyword = (filters or {}).get("keyword", "")
    if explicit_keyword:
        dec_filters = {**filters_base, "doc_types": ["uodo_decision"]}
        for d in keyword_exact_search(explicit_keyword, filters=dec_filters):
            _add(decisions, d)

    # ── 1b. Bezpośrednie frazy z zapytania → scroll po tagu ──────────────
    # Wyekstrahuj frazy 1- i 2-wyrazowe z query i szukaj ich jako tagów.
    # Robimy to BEZ LLM, żeby nie dostawać tagów pokrewnych (np. "dane
    # biometryczne") gdy użytkownik pyta o "dane genetyczne". LLM-tagi
    # (matched_tags) używamy tylko jako uzupełnienie gdy to nic nie daje.
    _stopwords = {
        "jakie",
        "są",
        "w",
        "o",
        "i",
        "z",
        "do",
        "na",
        "co",
        "ile",
        "jak",
        "czy",
        "przez",
        "dla",
        "po",
        "przy",
        "od",
        "ze",
        "to",
        "a",
        "że",
        "się",
        "nie",
        "być",
        "który",
        "które",
        "która",
    }
    _words = [
        w.lower()
        for w in re.split(r"\W+", query)
        if w.lower() not in _stopwords and len(w) > 2
    ]
    # Frazy: pojedyncze słowa + pary sąsiednich słów
    _direct_phrases: list[str] = list(
        dict.fromkeys(
            _words + [f"{_words[i]} {_words[i + 1]}" for i in range(len(_words) - 1)]
        )
    )
    all_tags_lower = {t.lower(): t for t in _get_all_tags()}
    _direct_hits: list[str] = []
    for phrase in _direct_phrases:
        if phrase in all_tags_lower:
            _direct_hits.append(all_tags_lower[phrase])

    for tag in _direct_hits:
        dec_filters = {**filters_base, "doc_types": ["uodo_decision"]}
        for d in keyword_exact_search(tag, filters=dec_filters):
            _add(decisions, d)

    # ── 1c. Matched tags (LLM) → tylko gdy 1b nic nie znalazło ──────────
    # Używamy tagów LLM wyłącznie jako fallback — LLM może zwrócić tagi
    # pokrewne ("dane biometryczne") które zaśmiecają wyniki dla konkretnej frazy.
    if not decisions and matched_tags:
        for tag in matched_tags:
            dec_filters = {**filters_base, "doc_types": ["uodo_decision"]}
            for d in keyword_exact_search(tag, filters=dec_filters):
                _add(decisions, d)

    # ── 1d. Semantic search — ostatni fallback gdy tagi nic nie dały ─────
    if len(decisions) < 5:
        dec_sem_filters = {**filters_base, "doc_types": ["uodo_decision"]}
        for d in semantic_search(
            query, top_k=20, filters=dec_sem_filters, score_threshold=0.45
        ):
            _add(decisions, d)

    # Posortuj decyzje malejąco po score
    decisions.sort(key=lambda d: -d.get("_score", 0))

    # ═══════════════════════════════════════════════════════════════════════
    # ARTYKUŁY u.o.d.o. — pomocnicze, max MAX_ACT_DOCS
    # ═══════════════════════════════════════════════════════════════════════

    # Explicit keyword z UI — scroll po tagu
    if explicit_keyword:
        act_filters = {**filters_base, "doc_types": ["legal_act_article"]}
        for d in keyword_exact_search(explicit_keyword, filters=act_filters):
            if len(act_docs) >= MAX_ACT_DOCS:
                break
            _add(act_docs, d)

    if len(act_docs) < MAX_ACT_DOCS:
        act_sem_filters = {**filters_base, "doc_types": ["legal_act_article"]}
        for d in semantic_search(
            query,
            top_k=MAX_ACT_DOCS - len(act_docs),
            filters=act_sem_filters,
            score_threshold=0.25,
        ):
            if len(act_docs) >= MAX_ACT_DOCS:
                break
            _add(act_docs, d)

    # ═══════════════════════════════════════════════════════════════════════
    # ARTYKUŁY RODO — kontekst prawny, max MAX_GDPR_DOCS
    # ═══════════════════════════════════════════════════════════════════════

    if explicit_keyword:
        gdpr_filters = {
            **filters_base,
            "doc_types": ["gdpr_article", "gdpr_recital"],
        }
        for d in keyword_exact_search(explicit_keyword, filters=gdpr_filters):
            if len(gdpr_docs) >= MAX_GDPR_DOCS:
                break
            _add(gdpr_docs, d)

    if len(gdpr_docs) < MAX_GDPR_DOCS:
        gdpr_sem_filters = {
            **filters_base,
            "doc_types": ["gdpr_article", "gdpr_recital"],
        }
        for d in semantic_search(
            query,
            top_k=MAX_GDPR_DOCS - len(gdpr_docs),
            filters=gdpr_sem_filters,
            score_threshold=0.3,
        ):
            if len(gdpr_docs) >= MAX_GDPR_DOCS:
                break
            _add(gdpr_docs, d)

    # ═══════════════════════════════════════════════════════════════════════
    # Złącz: decyzje pierwsze, potem u.o.d.o., potem RODO
    # ═══════════════════════════════════════════════════════════════════════
    merged = decisions + act_docs + gdpr_docs

    if not use_graph or not decisions:
        return merged, matched_tags

    # ── Graf — rozszerza tylko decyzje UODO ──────────────────────────────
    seed_sigs = [d.get("signature", "") for d in decisions if d.get("signature")]
    if seed_sigs:
        expanded = graph_expand(seed_sigs)
        seen_graph = {d.get("signature", "") for d in decisions}
        for sig, rel_type, score in expanded:
            if sig in seen_graph:
                continue
            doc = fetch_by_signature(sig)
            if doc:
                doc["_score"] = score
                doc["_graph_relation"] = rel_type
                decisions.append(doc)
                seen_graph.add(sig)
        # Odbuduj merged po rozszerzeniu grafu
        merged = decisions + act_docs + gdpr_docs

    return merged, matched_tags


# ─────────────────────────── LLM ─────────────────────────────────


# POPRAWKA 2: większy max_len i dokładniejszy krok przeszukiwania okna
# — poprzednio step=300 mógł minąć okno zawierające szukaną frazę,
#   szczególnie gdy decyzja jest długa a fraza pojawia się raz, np. na pozycji 2500.
def _extract_fragment(content: str, query: str, max_len: int = 2000) -> str:
    if not content or len(content) <= max_len:
        return content
    stopwords = {
        "jakie",
        "są",
        "w",
        "o",
        "i",
        "z",
        "do",
        "na",
        "co",
        "ile",
        "jak",
        "czy",
        "przez",
        "dla",
        "po",
        "przy",
        "od",
        "ze",
        "to",
    }
    keywords = [
        w.lower()
        for w in re.split(r"\W+", query)
        if w.lower() not in stopwords and len(w) > 2
    ]
    if not keywords:
        return content[:max_len]
    step = (
        150  # było 300 — dokładniejsze przeszukiwanie, mniej szans na pominięcie frazy
    )
    best_score, best_pos = -1, 0
    cl = content.lower()
    for pos in range(0, max(1, len(content) - max_len), step):
        score = sum(cl[pos : pos + max_len].count(kw) for kw in keywords)
        if score > best_score:
            best_score, best_pos = score, pos
    fragment = content[best_pos : best_pos + max_len]
    if best_pos > 0:
        nl = fragment.find("\n")
        if 0 < nl < 150:
            fragment = fragment[nl:].lstrip()
        fragment = "[…]\n" + fragment
    return fragment


# POPRAWKA 3: priorytetyzacja typów dokumentów w kontekście
# Kolejność: decyzje UODO → u.o.d.o. → RODO → reszta
# Bez tej zmiany artykuły RODO (wysoki score semantyczny dla "dane genetyczne")
# wypełniają limit max_chars zanim trafią decyzje UODO.
_CONTEXT_TYPE_ORDER = {
    "uodo_decision": 0,
    "legal_act_article": 1,
    "gdpr_article": 2,
    "gdpr_recital": 3,
}


def build_context(
    docs: list[dict[str, Any]],
    query: str,
    max_chars: int = 18000,
    filters: dict[str, Any] | None = None,
    memory: "AgentMemory | None" = None,
) -> str:
    """Buduje kontekst dla LLM używając szablonów Jinja2.
    Wzorzec z lekcji 1.2: Context Engineering — każdy typ dokumentu
    ma własny szablon co poprawia 'zakotwiczenie uwagi' modelu.
    Opcjonalnie wzbogaca kontekst o wpisy z pamięci epizodycznej (lekcja 2.1).
    """
    # ── Nota o filtrach ──────────────────────────────────────────
    f = filters or {}
    filter_lines = []
    if f.get("status"):
        filter_lines.append(f"Status decyzji: {f['status']}")
    if f.get("term_decision_type"):
        filter_lines.append(f"Rodzaj decyzji: {', '.join(f['term_decision_type'])}")
    if f.get("term_violation_type"):
        filter_lines.append(f"Rodzaj naruszenia: {', '.join(f['term_violation_type'])}")
    if f.get("term_legal_basis"):
        filter_lines.append(f"Podstawa prawna: {', '.join(f['term_legal_basis'])}")
    if f.get("term_corrective_measure"):
        filter_lines.append(
            f"Środek naprawczy: {', '.join(f['term_corrective_measure'])}"
        )
    if f.get("term_sector"):
        filter_lines.append(f"Sektor: {', '.join(f['term_sector'])}")
    if f.get("keyword"):
        filter_lines.append(f"Słowo kluczowe: {f['keyword']}")

    filter_note = ""
    if filter_lines:
        filter_note = (
            "UWAGA: Wyniki zawężone filtrami: "
            + "; ".join(filter_lines)
            + ". Odpowiadaj z uwzględnieniem tego kontekstu.\n"
        )

    # ── Nota z pamięci epizodycznej ──────────────────────────────
    memory_note = ""
    if memory:
        related = memory.find_related(query)
        if related:
            snippets = []
            for e in related[:2]:
                sigs = ", ".join(e.top_signatures[:3]) if e.top_signatures else "brak"
                snippets.append(
                    f"- Poprzednie pytanie: «{e.query}» → znalezione decyzje: {sigs}"
                )
            memory_note = (
                "KONTEKST Z POPRZEDNICH ANALIZ (tej sesji):\n"
                + "\n".join(snippets)
                + "\n"
            )

    # ── Nagłówek przez Jinja2 ────────────────────────────────────
    header = _TPL_HEADER.render(
        query=query,
        filter_note=filter_note,
        memory_note=memory_note,
    )
    parts = [header]
    chars = len(header)

    # ── Priorytetyzacja typów: decyzje UODO pierwsze, RODO ostatnie ──────
    # Bez tego sortowania artykuły RODO (wysoki score semantyczny) wypychają
    # decyzje UODO poza limit max_chars — duże modele nie widzą wtedy decyzji.
    docs_sorted = sorted(
        docs,
        key=lambda d: (
            _CONTEXT_TYPE_ORDER.get(d.get("doc_type", ""), 9),
            -d.get("_score", 0),
        ),
    )

    # ── Bloki dokumentów przez Jinja2 ────────────────────────────
    for i, doc in enumerate(docs_sorted, 1):
        dtype = doc.get("doc_type", "")

        if dtype == "legal_act_article":
            chunk_idx = doc.get("chunk_index", 0)
            total = doc.get("chunk_total", 1)
            suffix = f"(część {chunk_idx + 1}/{total})" if total > 1 else ""
            block = _TPL_ACT_ARTICLE.render(
                rank=i,
                art_num=doc.get("article_num", "?"),
                label_suffix=suffix,
                text=doc.get("content_text", ""),
            )
        elif dtype in ("gdpr_article", "gdpr_recital"):
            art_num = doc.get("article_num", "?")
            prefix = "Motyw" if dtype == "gdpr_recital" else f"Art. {art_num}"
            block = _TPL_GDPR.render(
                rank=i,
                prefix=prefix,
                text=doc.get("content_text", ""),
            )
        else:
            keywords = doc.get("keywords_text", "") or ", ".join(
                doc.get("keywords", [])
            )
            acts = doc.get("related_acts", [])[:4] + doc.get("related_eu_acts", [])[:2]
            block = _TPL_DECISION.render(
                rank=i,
                sig=doc.get("signature", "?"),
                date=doc.get("date_issued", "")[:7],
                status=doc.get("status", ""),
                graph_rel=doc.get("_graph_relation", ""),
                keywords=keywords[:200] if keywords else "",
                acts=", ".join(acts[:5]) if acts else "",
                fragment=_extract_fragment(doc.get("content_text", ""), query),
            )

        if chars + len(block) > max_chars:
            parts.append(f"\n[pominięto {len(docs_sorted) - i + 1} dalszych wyników]")
            break
        parts.append(block)
        chars += len(block)

    return "\n---\n".join(parts)


@st.cache_data(ttl=300, show_spinner=False)
def get_available_models(provider: str, api_key: str | None = None) -> list[str]:
    """Pobiera listę aktywnych modeli z API providera."""
    if provider == "Groq":
        try:
            from groq import Groq

            client = Groq(api_key=api_key or GROQ_API_KEY)
            models_resp = client.models.list()
            ids = sorted(
                m.id
                for m in models_resp.data
                if not any(x in m.id for x in ("whisper", "tts", "playai", "distil"))
            )
            return ids or ["llama-3.3-70b-versatile"]
        except Exception as e:
            st.warning(f"Nie udało się pobrać modeli Groq: {e}")
            return ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]

    # Ollama Cloud
    try:
        import requests as _req

        r = _req.get(
            f"{OLLAMA_CLOUD_URL}/api/tags",
            headers={"Authorization": f"Bearer {api_key or OLLAMA_CLOUD_API_KEY}"},
            timeout=10,
        )
        r.raise_for_status()
        models = [m.get("name") for m in r.json().get("models", []) if m.get("name")]
        return sorted(models) or ["qwen3:14b"]
    except Exception as e:
        st.warning(f"Nie udało się pobrać modeli Ollama Cloud: {e}")
        return ["qwen3:14b", "llama3.3:70b", "bielik:11b-v3"]


def call_llm_stream(
    query: str,
    context: str,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> Generator[str, Any, None]:
    """Stream odpowiedzi z Groq lub Ollama Cloud."""
    system = (
        "Jesteś ekspertem ds. ochrony danych osobowych i prawa RODO. "
        "Pomagasz analizować decyzje Prezesa UODO oraz przepisy ustawy o ochronie danych osobowych. "
        "Odpowiadaj po polsku, precyzyjnie i zwięźle. "
        "Zawsze powołuj się na konkretne decyzje UODO podając sygnatury [np. DKN.XXXX.XX.XXXX, ZSOŚS, i in.] "
        "lub artykuły ustawy [np. Art. X u.o.d.o.]. "
        "Jeśli kontekst nie zawiera odpowiedzi na pytanie, powiedz o tym wprost."
    )
    user = f"Pytanie: {query}\n\nDokumenty:\n{context}"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    provider = provider or st.session_state.get("llm_provider", DEFAULT_PROVIDER)
    model = model or st.session_state.get("llm_model", "")
    api_key = api_key or st.session_state.get("llm_api_key", "")

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

    elif provider == "Ollama Cloud":
        import json as _json

        import requests as _req

        resp = _req.post(
            f"{OLLAMA_CLOUD_URL}/api/chat",
            headers={"Authorization": f"Bearer {api_key or OLLAMA_CLOUD_API_KEY}"},
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
    else:
        yield "❌ Nieznany provider LLM."


# ─────────────────────────── REASONING STEP ──────────────────────
# Wzorzec z lekcji 4.1: LLM dekompozycja pytania PRZED wyszukiwaniem


def _call_llm_json(
    prompt: str,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Wywołanie LLM z wymaganym wyjściem JSON (bez streamowania)."""
    import json as _json

    provider = provider or st.session_state.get("llm_provider", DEFAULT_PROVIDER)
    model = model or st.session_state.get("llm_model", "")
    api_key = api_key or st.session_state.get("llm_api_key", "")
    messages = [
        {
            "role": "system",
            "content": "Odpowiadaj WYŁĄCZNIE poprawnym JSON. Bez komentarzy.",
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
            return _json.loads(resp.choices[0].message.content or "{}")
        elif provider == "Ollama Cloud":
            import requests as _req

            resp = _req.post(
                f"{OLLAMA_CLOUD_URL}/api/chat",
                headers={"Authorization": f"Bearer {api_key or OLLAMA_CLOUD_API_KEY}"},
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "format": "json",
                },
                timeout=30,
            )
            return _json.loads(resp.json().get("message", {}).get("content", "{}"))
    except Exception:
        pass
    return {}


def decompose_query(
    query: str,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> QueryDecomposition:
    """Reasoning Step — LLM analizuje pytanie i generuje ustrukturyzowane
    parametry wyszukiwania. Wzorzec z lekcji 4.1 kursu Software 3.0.

    Dla krótkich/prostych zapytań (< 10 słów) zwraca uproszczoną dekompozycję
    bez wywołania LLM, żeby nie spowalniać wyszukiwania.
    """
    # Szybka ścieżka dla prostych zapytań — nie wywołuj LLM
    words = query.strip().split()
    if len(words) <= 3:
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

    raw = _call_llm_json(prompt, provider=provider, model=model, api_key=api_key)
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


# ─────────────────────────── STATYSTYKI ──────────────────────────


@st.cache_data(ttl=3600)
def get_collection_stats() -> dict[str, Any]:
    client = get_qdrant()
    info = client.get_collection(COLLECTION_NAME)
    total = info.points_count

    decision_count = 0
    act_chunk_count = 0
    offset = None
    while True:
        pts, next_off = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=500,
            offset=offset,
            with_payload=["doc_type"],
            with_vectors=False,
        )
        for p in pts:
            dtype = (p.payload or {}).get("doc_type", "")
            if dtype == "uodo_decision":
                decision_count += 1
            elif dtype == "legal_act_article":
                act_chunk_count += 1
        if next_off is None:
            break
        offset = next_off

    G = get_graph()
    graph_stats = {}
    if G:
        uodo = [
            n for n, d in G.nodes(data=True) if d.get("doc_type") == "uodo_decision"
        ]
        most_cited = sorted(
            [(n, G.in_degree(n)) for n in uodo if G.in_degree(n) > 0],
            key=lambda x: -x[1],
        )[:5]
        graph_stats = {
            "edges": G.number_of_edges(),
            "most_cited": most_cited,
        }

    return {
        "total": total,
        "decisions": decision_count,
        "act_chunks": act_chunk_count,
        **graph_stats,
    }


# ─────────────────────────── KARTY WYNIKÓW ───────────────────────


def decision_url(doc: dict[str, Any]) -> str:
    sig = doc.get("signature", "")
    url = doc.get("source_url", "")
    if url:
        return url
    slug = sig.lower().replace(".", "_")
    year_m = re.search(r"\b(20\d{2})\b", sig)
    year = year_m.group(1) if year_m else "2024"
    return f"{UODO_PORTAL_BASE}/urn:ndoc:gov:pl:uodo:{year}:{slug}/content"


def render_act_article_card(doc: dict[str, Any], rank: int):
    """Karta dla artykułu ustawy o ochronie danych osobowych."""
    art_num = doc.get("article_num", "?")
    chunk_idx = doc.get("chunk_index", 0)
    total = doc.get("chunk_total", 1)
    score = doc.get("_score", 0)
    text = doc.get("content_text", "")[:600]

    label = f"Art. {art_num} u.o.d.o."
    if total > 1:
        label += f" (część {chunk_idx + 1}/{total})"

    html = f"""
    <article class="doc-list-item">
      <header>
        <a href="{ISAP_ACT_URL}" target="_blank">{label}</a>
        <span><small>Ustawa o ochronie danych osobowych</small></span>
      </header>
      <main>
        <h2><a href="{ISAP_ACT_URL}" target="_blank">Dz.U. 2019 poz. 1781</a>
          <span class="status-badge status-final ms-2">u.o.d.o.</span>
        </h2>
        <p>{text}{"…" if len(doc.get("content_text", "")) > 600 else ""}</p>
      </main>
      <footer>
        <small class="text-muted">score: {score:.3f}</small>
      </footer>
    </article>"""
    st.markdown(html, unsafe_allow_html=True)


def render_decision_card(doc: dict[str, Any], rank: int):
    """Karta dla decyzji UODO."""
    sig = doc.get("signature", "?")
    status = doc.get("status", "")
    date = doc.get("date_published", "") or doc.get("date_issued", "")
    source = doc.get("_source", "")
    graph_rel = doc.get("_graph_relation", "")
    title = doc.get("title_full", "") or doc.get("title", "")
    name = doc.get("title", sig)
    url = decision_url(doc)

    kw_list = doc.get("keywords", [])
    if isinstance(kw_list, str):
        kw_list = [k.strip() for k in kw_list.split(",") if k.strip()]

    kinds = doc.get("term_decision_type", [])
    sectors = doc.get("term_sector", [])

    # Usuń z keywords to, co już wyświetlamy jako kinds/sectors (unikamy duplikatów)
    taxonomy_values = {v.lower() for v in kinds + sectors}
    kw_list = [k for k in kw_list if k.lower() not in taxonomy_values]

    status_cls = {
        "prawomocna": "status-final",
        "nieprawomocna": "status-nonfinal",
        "uchylona": "status-repealed",
    }.get(status, "status-unknown")

    date_fmt = ""
    if date:
        try:
            from datetime import datetime

            d = datetime.strptime(date[:10], "%Y-%m-%d")
            months = [
                "stycznia",
                "lutego",
                "marca",
                "kwietnia",
                "maja",
                "czerwca",
                "lipca",
                "sierpnia",
                "września",
                "października",
                "listopada",
                "grudnia",
            ]
            date_fmt = f"{d.day} {months[d.month - 1]} {d.year}"
        except Exception:
            date_fmt = date[:10]

    graph_badge = ""
    if source == "graph":
        graph_badge = (
            f' <span class="status-badge status-unknown">↗ {graph_rel or "graf"}</span>'
        )

    # ── Nagłówek karty (HTML — styl portalu)
    st.markdown(
        f"""
    <article class="doc-list-item">
      <header>
        <a href="{url}" target="_blank">{sig}</a>
        <time><small>opublikowano</small> {date_fmt}</time>
      </header>
      <main>
        <h2 class="d-flex justify-content-between align-items-start gap-2">
          <a href="{url}" target="_blank">{name}</a>
          <span class="status-badge {status_cls}">{status.upper()}{graph_badge}</span>
        </h2>
        <p class="text-muted">{title[:280] + "…" if len(title) > 280 else title}</p>
      </main>
    </article>""",
        unsafe_allow_html=True,
    )

    # ── Footer karty — Streamlit (żeby markdown działał poprawnie)
    with st.container():
        # Słowa kluczowe
        if kw_list:
            shown = kw_list[:8]
            rest = len(kw_list) - len(shown)
            tags = " · ".join(f"`{k}`" for k in shown)
            suffix = f" *+{rest} więcej*" if rest > 0 else ""
            st.caption(f"🏷️ {tags}{suffix}")

        # Powołane akty
        all_acts = doc.get("related_acts", [])[:4] + doc.get("related_eu_acts", [])[:2]
        if all_acts:
            st.caption("📜 Powołane akty: " + " · ".join(f"`{a}`" for a in all_acts))

        if graph_rel:
            st.caption(f"↗ powiązana przez graf: *{graph_rel}*")

    st.divider()


def render_gdpr_card(doc: dict[str, Any], rank: int):
    """Karta dla artykułu lub motywy RODO."""
    art_num = doc.get("article_num", "?")
    chunk_idx = doc.get("chunk_index", 0)
    total = doc.get("chunk_total", 1)
    score = doc.get("_score", 0)
    text = doc.get("content_text", "")[:500]
    dtype = doc.get("doc_type", "")
    chapter = doc.get("chapter", "")
    chapter_title = doc.get("chapter_title", "")

    is_recital = dtype == "gdpr_recital"
    label = art_num if is_recital else f"Art. {art_num} RODO"
    badge_txt = "motyw RODO" if is_recital else "RODO"
    if not is_recital and total > 1:
        label += f" (część {chunk_idx + 1}/{total})"

    chapter_html = ""
    if chapter and chapter_title:
        chapter_html = (
            f'<small class="text-muted">Rozdział {chapter} — {chapter_title}</small>'
        )

    html = f"""
    <article class="doc-list-item">
      <header>
        <a href="{GDPR_URL}" target="_blank">{label}</a>
        <span class="status-badge status-final">{badge_txt}</span>
      </header>
      <main>
        <h2>{chapter_html}</h2>
        <p>{text}{"…" if len(doc.get("content_text", "")) > 500 else ""}</p>
      </main>
      <footer>
        <small class="text-muted">score: {score:.3f}</small>
      </footer>
    </article>"""
    st.markdown(html, unsafe_allow_html=True)


def render_card(doc: dict[str, Any], rank: int):
    """Dispatcher — wybiera typ karty na podstawie doc_type."""
    dtype = doc.get("doc_type", "")
    if dtype == "legal_act_article":
        render_act_article_card(doc, rank)
    elif dtype in ("gdpr_article", "gdpr_recital"):
        render_gdpr_card(doc, rank)
    else:
        render_decision_card(doc, rank)


# ─────────────────────────── GŁÓWNA APLIKACJA ────────────────────


def main():
    st.set_page_config(
        page_title="Portal Orzeczeń UODO — Wyszukiwarka",
        page_icon="🔐",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.markdown(
        """
    <style>
        /* ── Red Hat Display — font portalu UODO ── */
        @import url('https://fonts.googleapis.com/css2?family=Red+Hat+Display:wght@400;500;600;700;800&display=swap');

        /* ── Zmienne CSS z root.css portalu UODO ── */
        :root {
            --uodo-blue-10: #f5f8f8;
            --uodo-blue-20: #e8f1fd;
            --uodo-blue-30: #dde3ee;
            --uodo-blue-33: #a5b3dd;
            --uodo-blue-35: #6d83cc;
            --uodo-blue-38: #356bcc;
            --uodo-blue-40: #0058cc;
            --uodo-blue-50: #275faa;
            --uodo-blue-60: #0e4591;
            --uodo-blue-80: #092e60;
            --uodo-dark-gray: #3f444f;
            --uodo-light-gray: #c8ccd3;
            --uodo-red: #f25a5a;
            --uodo-red-logo: #cd071e;
            --uodo-red-dark: #b22222;
            --uodo-white: #fff;
            --uodo-black: rgba(26,26,28,1);
            --body-color: rgba(26,26,28,1);
            --content-width: 1070px;
            --link-color: var(--uodo-blue-60);
            --link-hover-color: var(--uodo-blue-40);
            --separator-color: var(--uodo-blue-30);
            --sidebar-bgcolor: var(--uodo-blue-10);
            --uodo-border-radius: 2px;
        }

        /* ── Typografia ── */
        html, body, [class*="css"] {
            font-family: 'Red Hat Display', sans-serif !important;
            color: var(--body-color);
        }

        /* ── Ukryj elementy Streamlit ── */
        [data-testid="stHeader"]  { display: none; }
        footer                    { display: none; }
        .main .block-container    { padding-top: 0 !important; max-width: 1150px; }

        /* ── Sidebar ── */
        [data-testid="stSidebar"] { background: var(--uodo-blue-80); }
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stMarkdown,
        [data-testid="stSidebar"] p { color: #c5d3e8 !important; }
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 { color: white !important; }
        [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.15); }

        /* ── Nagłówek strony — wzór page-header z header.css ── */
        .page-header {
            padding: 20px 0 16px;
            box-shadow: 0 5px 20px rgba(14,69,145,0.07);
            margin: -1rem -1rem 1.5rem -1rem;
            background: var(--uodo-white);
            border-bottom: 1px solid var(--uodo-blue-30);
        }
        .page-header-inner {
            max-width: var(--content-width);
            margin: 0 auto;
            padding: 0 2rem;
            display: flex;
            align-items: center;
            gap: 1.5rem;
        }
        .page-header h1 {
            color: var(--uodo-red-logo);
            font-size: 1.8rem;
            font-weight: 800;
            margin: 0;
            letter-spacing: -0.01em;
        }
        .page-header-sub {
            color: var(--uodo-dark-gray);
            font-size: 0.85rem;
            margin: 2px 0 0;
        }

        /* ── Formularz wyszukiwania — featured-card z app.css ── */
        .featured-card {
            background-color: var(--uodo-blue-20);
            padding: 2rem 2.5rem;
            border-radius: var(--uodo-border-radius);
            margin-bottom: 1.5rem;
        }

        /* ── Przyciski ── */
        .stButton > button[kind="primary"] {
            background-color: var(--uodo-blue-60) !important;
            border-color: var(--uodo-blue-60) !important;
            color: white !important;
            font-family: 'Red Hat Display', sans-serif !important;
            font-weight: 600 !important;
            border-radius: var(--uodo-border-radius) !important;
            transition: background-color 200ms !important;
        }
        .stButton > button[kind="primary"]:hover {
            background-color: var(--uodo-blue-50) !important;
            border-color: var(--uodo-blue-50) !important;
        }

        /* ── Karta wyniku — doc-list-item z app.css ── */
        article.doc-list-item {
            border: 1px solid var(--uodo-blue-30);
            border-radius: var(--uodo-border-radius);
            margin-bottom: 24px;
            font-family: 'Red Hat Display', sans-serif;
        }
        article.doc-list-item > header {
            background-color: var(--uodo-blue-10);
            padding: 10px 20px;
            display: flex;
            flex-direction: row;
            justify-content: space-between;
            align-items: center;
            border-radius: var(--uodo-border-radius) var(--uodo-border-radius) 0 0;
            transition: background-color 200ms;
        }
        article.doc-list-item > header > a {
            color: var(--uodo-blue-60);
            font-weight: 600;
            font-size: 1.1rem;
            text-decoration: none;
        }
        article.doc-list-item > header time {
            color: var(--uodo-dark-gray);
            font-size: 0.85rem;
        }
        article.doc-list-item:hover > header {
            background-color: var(--uodo-blue-50);
            transition: background-color 200ms;
        }
        article.doc-list-item:hover > header > a,
        article.doc-list-item:hover > header time,
        article.doc-list-item:hover > header small {
            color: var(--uodo-white) !important;
        }
        article.doc-list-item > main {
            color: var(--uodo-dark-gray);
            padding: 0 20px;
        }
        article.doc-list-item > main > * {
            display: block;
            margin-bottom: 16px;
        }
        article.doc-list-item > main > *:first-child {
            margin-top: 16px;
        }
        article.doc-list-item > main h2 {
            font-weight: 700;
            font-size: 1rem;
            line-height: 150%;
            color: var(--uodo-dark-gray);
            margin: 0 0 8px;
        }
        article.doc-list-item > main h2 a {
            color: var(--uodo-dark-gray);
            text-decoration: none;
        }
        article.doc-list-item > main h2 a:hover {
            color: var(--uodo-blue-40);
        }
        article.doc-list-item > main a {
            color: var(--uodo-dark-gray);
            font-size: 0.92rem;
            text-decoration: none;
        }
        article.doc-list-item > main p {
            margin: 0;
            font-size: 0.92rem;
        }
        article.doc-list-item > footer {
            margin: 0 20px;
            border-top: 1px solid var(--uodo-blue-30);
            padding: 12px 0 14px;
            overflow: hidden;
        }

        /* ── Badge statusu ── */
        .status-badge {
            display: inline-block;
            padding: 2px 10px;
            border-radius: 2px;
            font-size: 0.75rem;
            font-weight: 600;
            white-space: nowrap;
        }
        .status-final       { background: #d1fae5; color: #065f46; }
        .status-nonfinal    { background: #dbeafe; color: #1e40af; }
        .status-repealed    { background: #f3f4f6; color: #374151; }
        .status-unknown     { background: #fef9c3; color: #713f12; }

        /* ── Tagi wyników — ui-result-tags z app.css ── */
        .ui-result-tags {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 4px;
            font-size: 0.78rem;
            color: var(--uodo-blue-60);
            padding: 4px 0 0;
        }
        .ui-result-tag {
            padding: 1px 8px;
            border-right: 1px solid var(--uodo-blue-30);
            line-height: 1.5;
        }
        .ui-result-tag:last-child { border-right: none; }

        /* ── Odpowiedź AI ── */
        .answer-box {
            background: var(--uodo-blue-10);
            border-left: 4px solid var(--uodo-blue-60);
            padding: 1rem 1.2rem;
            border-radius: 2px;
            margin: 0.5rem 0 1rem;
            font-family: 'Red Hat Display', sans-serif;
        }

        /* ── Linki globalne ── */
        a { color: var(--link-color); text-decoration: none; }
        a:hover { color: var(--link-hover-color); }

        /* ── Taby ── */
        [data-testid="stTabs"] [data-baseweb="tab"] {
            font-family: 'Red Hat Display', sans-serif !important;
            font-size: 0.88rem !important;
        }
        [data-testid="stTabs"] [aria-selected="true"] {
            color: var(--uodo-blue-60) !important;
            border-bottom-color: var(--uodo-blue-60) !important;
        }

        /* ── Expander ── */
        div[data-testid="stExpander"] {
            border: 1px solid var(--uodo-blue-30) !important;
            border-radius: var(--uodo-border-radius) !important;
        }

        /* ── Keyword toggle ── */
        .kw-toggle:hover { text-decoration: underline; }
        .acts-row {
            margin-top: 6px;
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 4px;
            font-size: 0.78rem;
        }

        /* ── Pasek wyszukiwania z ikoną filtrów ── */
        .search-bar-wrap {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 0;
        }
        .filter-toggle-btn {
            background: var(--uodo-blue-60);
            border: none;
            border-radius: 2px;
            color: white;
            padding: 8px 14px;
            cursor: pointer;
            font-size: 1rem;
            display: flex;
            align-items: center;
            gap: 6px;
            transition: background 200ms;
        }
        .filter-toggle-btn:hover { background: var(--uodo-blue-50); }
        .filter-toggle-btn.active { background: var(--uodo-blue-80); }
        .filters-panel {
            background: var(--uodo-blue-20);
            border: 1px solid var(--uodo-blue-30);
            border-radius: 2px;
            padding: 1.2rem 1.5rem 1rem;
            margin-top: 0.75rem;
            border-top: 2px solid var(--uodo-blue-60);
        }
        .filter-label {
            font-size: 0.78rem;
            font-weight: 700;
            color: var(--uodo-blue-80);
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 4px;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # ── Nagłówek portalu ────────────────────────────────────────
    st.markdown(
        """
    <div class="page-header">
      <div class="page-header-inner">
        <div>
          <h1>Portal Orzeczeń UODO</h1>
          <div class="page-header-sub">Wyszukiwarka decyzji Prezesa UODO · Ustawa o ochronie danych osobowych · RODO</div>
        </div>
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # ── Sidebar — opcje techniczne ──────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Opcje")

        provider = st.selectbox(
            "Provider LLM", ["Ollama Cloud", "Groq"], key="provider_select"
        )
        api_key = st.text_input(
            "Klucz API",
            type="password",
            value=st.session_state.get("llm_api_key", ""),
            key="api_key_input",
        )

        models = get_available_models(provider, api_key)

        default_model = (
            DEFAULT_OLLAMA_MODEL if provider == "Ollama Cloud" else DEFAULT_GROQ_MODEL
        )
        default_idx = next((i for i, m in enumerate(models) if default_model in m), 0)
        selected_model = st.selectbox("Model", models, index=default_idx)

        st.session_state["llm_provider"] = provider
        st.session_state["llm_model"] = selected_model
        st.session_state["llm_api_key"] = api_key

        st.markdown("---")
        use_graph = st.toggle("Graf powiązań", value=True)

        st.markdown("### 📂 Typ dokumentów")
        show_decisions = st.checkbox("Decyzje UODO", value=True)
        show_act = st.checkbox("Ustawa o ochronie danych (u.o.d.o.)", value=True)
        show_gdpr = st.checkbox("RODO (rozporządzenie UE 2016/679)", value=True)

        st.markdown("---")
        try:
            stats = get_collection_stats()
            st.markdown("### 📊 Baza wiedzy")
            st.metric("Decyzje UODO", stats.get("decisions", 0))
            st.metric("Artykuły u.o.d.o.", stats.get("act_chunks", 0))
            if stats.get("edges"):
                st.metric("Powiązania w grafie", stats.get("edges", 0))
        except Exception:
            pass

        # ── Historia sesji (pamięć epizodyczna — lekcja 2.1) ─────────────
        if "agent_memory" in st.session_state:
            mem: AgentMemory = st.session_state["agent_memory"]
            if mem.entries:
                st.markdown("---")
                st.markdown("### 🧠 Historia sesji")
                for i, e in enumerate(mem.entries):
                    short_q = e.query[:40] + ("…" if len(e.query) > 40 else "")
                    with st.expander(f"{i + 1}. {short_q}", expanded=False):
                        if e.decomposition_summary:
                            st.caption(f"_{e.decomposition_summary}_")
                        if e.top_signatures:
                            st.caption(
                                "📋 "
                                + " · ".join(f"`{s}`" for s in e.top_signatures[:3])
                            )
                        if e.top_articles:
                            st.caption("📜 " + " · ".join(e.top_articles))
                        if e.answer_snippet:
                            st.caption(e.answer_snippet[:200] + "…")

    # ── Filtry ──────────────────────────────────────────────────
    doc_types = []
    if show_decisions:
        doc_types.append("uodo_decision")
    if show_act:
        doc_types.append("legal_act_article")
    if show_gdpr:
        doc_types.extend(["gdpr_article", "gdpr_recital"])
    if not doc_types:
        doc_types = [
            "uodo_decision",
            "legal_act_article",
            "gdpr_article",
            "gdpr_recital",
        ]

    taxonomy = _get_taxonomy_options()

    # ── Sekcja wyszukiwania ─────────────────────────────────────
    if "_example_query" in st.session_state:
        st.session_state["query_input"] = st.session_state.pop("_example_query")

    col_q, col_ai, col_btn = st.columns([7, 1.5, 1.2])
    with col_q:
        query = st.text_input(
            "Treść",
            placeholder="Wpisz treść, sygnaturę lub temat...",
            key="query_input",
            label_visibility="collapsed",
        )
    with col_ai:
        use_llm = st.checkbox("🤖 Użyj AI", value=True, key="use_llm_cb")
    with col_btn:
        search_btn = st.button("🔍 Szukaj", type="primary", use_container_width=True)

    with st.expander("🔽 Filtry zaawansowane", expanded=False):
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            st.markdown(
                '<div class="filter-label">Sygnatura</div>', unsafe_allow_html=True
            )
            sig_filter = st.text_input(
                "Sygnatura",
                placeholder="np. DKN.5110",
                label_visibility="collapsed",
                key="sig_filter",
            )
            st.markdown(
                '<div class="filter-label">Status</div>', unsafe_allow_html=True
            )
            status_filter = st.selectbox(
                "Status",
                ["— wszystkie —", "prawomocna", "nieprawomocna", "uchylona"],
                label_visibility="collapsed",
                key="status_filter",
            )
            st.markdown(
                '<div class="filter-label">Słowa kluczowe</div>', unsafe_allow_html=True
            )
            all_tags = _get_all_tags()
            kw_filter = (
                st.selectbox(
                    "Słowo kluczowe",
                    options=[""] + all_tags,
                    label_visibility="collapsed",
                    key="kw_filter",
                )
                or ""
            )
        with fc2:
            st.markdown(
                '<div class="filter-label">Rodzaj decyzji</div>', unsafe_allow_html=True
            )
            tax_decision = st.multiselect(
                "Rodzaj decyzji",
                options=taxonomy.get("term_decision_type", []),
                label_visibility="collapsed",
                key="tax_decision",
            )
            st.markdown(
                '<div class="filter-label">Środek naprawczy</div>',
                unsafe_allow_html=True,
            )
            tax_measure = st.multiselect(
                "Środek naprawczy",
                options=taxonomy.get("term_corrective_measure", []),
                label_visibility="collapsed",
                key="tax_measure",
            )
            st.markdown(
                '<div class="filter-label">Podstawa prawna</div>',
                unsafe_allow_html=True,
            )
            tax_legal_basis = st.multiselect(
                "Podstawa prawna",
                options=taxonomy.get("term_legal_basis", []),
                label_visibility="collapsed",
                key="tax_legal_basis",
            )
        with fc3:
            st.markdown(
                '<div class="filter-label">Rodzaj naruszenia</div>',
                unsafe_allow_html=True,
            )
            tax_violation = st.multiselect(
                "Rodzaj naruszenia",
                options=taxonomy.get("term_violation_type", []),
                label_visibility="collapsed",
                key="tax_violation",
            )
            st.markdown(
                '<div class="filter-label">Sektor</div>', unsafe_allow_html=True
            )
            tax_sector = st.multiselect(
                "Sektor",
                options=taxonomy.get("term_sector", []),
                label_visibility="collapsed",
                key="tax_sector",
            )
            st.markdown(
                '<div class="filter-label">Data ogłoszenia (od–do)</div>',
                unsafe_allow_html=True,
            )
            dcol1, dcol2 = st.columns(2)
            with dcol1:
                date_from = st.text_input(
                    "Od",
                    placeholder="2020-01-01",
                    label_visibility="collapsed",
                    key="date_from",
                )
            with dcol2:
                date_to = st.text_input(
                    "Do",
                    placeholder="2026-12-31",
                    label_visibility="collapsed",
                    key="date_to",
                )

    # ── Przykładowe pytania ─────────────────────────────────────
    with st.expander("💡 Przykładowe pytania", expanded=not bool(query)):
        st.caption("Kliknij pytanie aby je wyszukać:")
        examples = [
            ("🔔", "Kiedy wymagane jest zgłoszenie naruszenia danych?"),
            ("⚖️", "Jakie kary może nałożyć Prezes UODO?"),
            ("🔐", "Brak podstawy prawnej przetwarzania danych"),
            ("✅", "Zgoda na przetwarzanie danych osobowych"),
            ("🧬", "Dane genetyczne"),
            ("🗳️", "Dane osobowe w kampanii wyborczej"),
            ("📋", "Obowiązek informacyjny administratora"),
            ("🤝", "Umowa powierzenia przetwarzania danych"),
            ("🕵️", "Inspektor ochrony danych — konflikt interesów"),
            ("📸", "Zdjęcie tablicy rejestracyjnej w internecie a RODO"),
            ("📜", "DKN.5131.15.2025"),
        ]
        cols = st.columns(2)
        for idx, (emoji, question) in enumerate(examples):
            col = cols[idx % 2]
            if col.button(
                f"{emoji} {question}", key=f"example_{idx}", use_container_width=True
            ):
                st.session_state["_example_query"] = question
                st.rerun()

    # ── Budowanie filtrów ───────────────────────────────────────
    filters = {"doc_types": doc_types}

    # Filtry dotyczące tylko decyzji UODO
    if "uodo_decision" in doc_types:
        if status_filter != "— wszystkie —":
            filters["status"] = status_filter
        if tax_decision:
            filters["term_decision_type"] = tax_decision
        if tax_violation:
            filters["term_violation_type"] = tax_violation
        if tax_legal_basis:
            filters["term_legal_basis"] = tax_legal_basis
        if tax_measure:
            filters["term_corrective_measure"] = tax_measure
        if tax_sector:
            filters["term_sector"] = tax_sector
        if date_from.strip():
            try:
                filters["year_from"] = int(date_from.strip()[:4])
            except ValueError:
                pass
        if date_to.strip():
            try:
                filters["year_to"] = int(date_to.strip()[:4])
            except ValueError:
                pass

    # Filtr słowa kluczowego i sygnatury — dla wszystkich typów
    if kw_filter.strip():
        filters["keyword"] = kw_filter.strip()

    # ── Inicjalizacja pamięci epizodycznej (lekcja 2.1 Memory Engineering) ──
    if "agent_memory" not in st.session_state:
        st.session_state["agent_memory"] = AgentMemory()
    memory: AgentMemory = st.session_state["agent_memory"]

    # ── Wyszukiwanie ────────────────────────────────────────────
    effective_query = query
    if sig_filter.strip() and not query.strip():
        effective_query = sig_filter.strip()

    if effective_query and (
        search_btn
        or st.session_state.get("last_query") != effective_query
        or st.session_state.get("last_filters") != str(filters)
    ):
        st.session_state["last_query"] = effective_query
        st.session_state["last_filters"] = str(filters)

        # ── Reasoning Step (lekcja 4.1) — dekompozycja PRZED wyszukiwaniem ──
        decomp: QueryDecomposition | None = None
        if use_llm and len(effective_query.split()) > 3:
            with st.spinner("🧠 Analizuję pytanie..."):
                decomp = decompose_query(effective_query)
            if decomp and decomp.reasoning:
                with st.expander(
                    "🧠 Reasoning Step — jak zrozumiałem pytanie", expanded=False
                ):
                    st.caption(f"**Typ zapytania:** {decomp.query_type.value}")
                    st.caption(f"**Rozumowanie:** {decomp.reasoning}")
                    if decomp.search_keywords:
                        st.caption(
                            "**Słowa kluczowe:** "
                            + " · ".join(f"`{k}`" for k in decomp.search_keywords)
                        )
                    if decomp.gdpr_articles_hint:
                        st.caption(
                            "**Wskazane artykuły RODO:** "
                            + ", ".join(decomp.gdpr_articles_hint)
                        )
                    if decomp.uodo_act_articles_hint:
                        st.caption(
                            "**Wskazane artykuły u.o.d.o.:** "
                            + ", ".join(decomp.uodo_act_articles_hint)
                        )
                    if decomp.enriched_query != effective_query:
                        st.caption(
                            f"**Wzbogacone zapytanie:** _{decomp.enriched_query}_"
                        )

        # Użyj enriched_query do wyszukiwania jeśli dekompozycja zadziałała
        search_query = decomp.enriched_query if decomp else effective_query

        # Wzbogać filtry o podpowiedzi z dekompozycji
        if decomp and decomp.year_from_hint and "year_from" not in filters:
            filters["year_from"] = decomp.year_from_hint
        if decomp and decomp.year_to_hint and "year_to" not in filters:
            filters["year_to"] = decomp.year_to_hint

        with st.spinner("🔍 Wyszukuję..."):
            t0 = time.time()
            _tags: list[str] = []
            sig_match = _RE_QUERY_SIG.match(effective_query)
            if sig_match:
                sig_norm = sig_match.group(1).upper()
                exact = fetch_by_signature(sig_norm)
                if exact:
                    exact["_source"] = "exact"
                    exact["_score"] = 1.0
                    docs = [exact]
                    if use_graph:
                        for rsig in exact.get("related_uodo_rulings", [])[:5]:
                            rdoc = fetch_by_signature(rsig)
                            if rdoc:
                                rdoc["_source"] = "graph"
                                rdoc["_score"] = 0.9
                                docs.append(rdoc)
                else:
                    st.warning(
                        f"Nie znaleziono decyzji o sygnaturze **{sig_norm}** w bazie."
                    )
                    docs, _tags = hybrid_search(
                        search_query,
                        top_k=TOP_K,
                        filters=filters,
                        use_graph=use_graph,
                    )
            else:
                docs, _tags = hybrid_search(
                    search_query, top_k=TOP_K, filters=filters, use_graph=use_graph
                )
            search_time = time.time() - t0

        if not docs:
            st.warning(
                "Nie znaleziono dokumentów. Spróbuj zmienić filtry lub sformułowanie."
            )
            return

        decisions = [d for d in docs if d.get("doc_type") == "uodo_decision"]
        act_arts = [d for d in docs if d.get("doc_type") == "legal_act_article"]
        gdpr_docs = [
            d for d in docs if d.get("doc_type") in ("gdpr_article", "gdpr_recital")
        ]
        graph_docs = [d for d in docs if d.get("_source") == "graph"]

        _tag_info = f" · tag: `{kw_filter}`" if kw_filter.strip() else ""
        st.caption(
            f"Znaleziono {len(docs)} dokumentów "
            f"({len(decisions)} decyzji, {len(act_arts)} u.o.d.o., "
            f"{len(gdpr_docs)} RODO, {len(graph_docs)} przez graf) · {search_time:.2f}s"
            + _tag_info
        )
        if _tags:
            st.caption("🏷️ Tagi: " + " · ".join(f"`{t}`" for t in _tags))

        if use_llm:
            context = build_context(
                docs, effective_query, filters=filters, memory=memory
            )
            st.markdown("### 💬 Odpowiedź AI")
            answer_placeholder = st.empty()
            full_answer = ""
            try:
                for chunk in call_llm_stream(effective_query, context):
                    full_answer += chunk
                    answer_placeholder.markdown(
                        f'<div class="answer-box">{full_answer}</div>',
                        unsafe_allow_html=True,
                    )
            except Exception as e:
                st.error(f"Błąd LLM: {e}")

            # ── Zapisz do pamięci epizodycznej (lekcja 2.1) ─────────────────
            if full_answer:
                top_sigs = [
                    d.get("signature", "") for d in decisions[:5] if d.get("signature")
                ]
                top_arts = [
                    f"Art. {d.get('article_num')}"
                    for d in act_arts[:3]
                    if d.get("article_num")
                ]
                memory.add(
                    MemoryEntry(
                        query=effective_query,
                        enriched_query=search_query,
                        decomposition_summary=decomp.reasoning if decomp else "",
                        top_signatures=top_sigs,
                        top_articles=top_arts,
                        answer_snippet=full_answer[:300],
                    )
                )

        st.markdown(f"### 📋 Dokumenty ({len(docs)})")
        tabs = st.tabs(
            [
                f"Wszystkie ({len(docs)})",
                f"Decyzje UODO ({len(decisions)})",
                f"Ustawa u.o.d.o. ({len(act_arts)})",
                f"RODO ({len(gdpr_docs)})",
                f"Graf ({len(graph_docs)})",
            ]
        )

        with tabs[0]:
            for i, doc in enumerate(docs, 1):
                render_card(doc, i)

        with tabs[1]:
            if decisions:
                for i, doc in enumerate(decisions, 1):
                    render_decision_card(doc, i)
            else:
                st.info("Brak decyzji UODO dla tego zapytania.")

        with tabs[2]:
            if act_arts:
                for i, doc in enumerate(act_arts, 1):
                    render_act_article_card(doc, i)
            else:
                st.info("Brak artykułów ustawy dla tego zapytania.")

        with tabs[3]:
            if gdpr_docs:
                for i, doc in enumerate(gdpr_docs, 1):
                    render_gdpr_card(doc, i)
            else:
                st.info("Brak artykułów RODO dla tego zapytania.")

        with tabs[4]:
            if graph_docs:
                st.info("Decyzje powiązane przez cytowania z wynikami semantic search.")
                for i, doc in enumerate(graph_docs, 1):
                    render_decision_card(doc, i)
            else:
                st.info("Brak wyników z grafu powiązań.")


if __name__ == "__main__":
    main()
