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
from typing import Dict, List, Optional, Tuple

import networkx as nx
import streamlit as st
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

    load_dotenv()
except ImportError:
    pass

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OLLAMA_CLOUD_API_KEY = os.getenv("OLLAMA_CLOUD_API_KEY", "")
OLLAMA_CLOUD_URL = os.getenv("OLLAMA_CLOUD_URL", "https://ollama.com")
OLLAMA_LOCAL_URL = os.getenv("OLLAMA_LOCAL_URL", "http://localhost:11434")

PROVIDERS = ["Ollama Cloud", "Groq"]
DEFAULT_PROVIDER = "Ollama Cloud"
DEFAULT_OLLAMA_MODEL = "gpt-oss:120b"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"

TOP_K = 8
GRAPH_DEPTH = 2
UODO_PORTAL_BASE = "https://orzeczenia.uodo.gov.pl/document"
ISAP_ACT_URL = "https://isap.sejm.gov.pl/isap.nsf/DocDetails.xsp?id=WDU20190001781"
GDPR_URL = "https://eur-lex.europa.eu/legal-content/PL/TXT/?uri=CELEX:32016R0679"


# ─────────────────────────── CACHE / ZASOBY ──────────────────────


@st.cache_resource
def get_qdrant() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, timeout=30)


@st.cache_resource
def get_embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBED_MODEL, trust_remote_code=True)


@st.cache_resource
def get_graph() -> Optional[nx.DiGraph]:
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


def embed(text: str) -> List[float]:
    return get_embedder().encode(text, normalize_embeddings=True).tolist()


def semantic_search(query: str, top_k: int = TOP_K, filters: Dict = None) -> List[Dict]:
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
        for term_field in ("term_decision_type", "term_violation_type",
                           "term_legal_basis", "term_corrective_measure", "term_sector"):
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
        score_threshold=0.25,
    )

    docs = []
    for hit in res.points or []:
        d = (hit.payload or {}).copy()
        d["_score"] = hit.score
        d["_source"] = "semantic"
        docs.append(d)
    return docs


def graph_expand(
    seed_sigs: List[str], depth: int = GRAPH_DEPTH
) -> List[Tuple[str, str, float]]:
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


def fetch_by_signature(sig: str) -> Optional[Dict]:
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


def keyword_exact_search(keyword: str, filters: Dict = None) -> List[Dict]:
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

    qdrant_filter = Filter(must=must)
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


@st.cache_data(ttl=300, show_spinner=False)
@st.cache_data(ttl=3600)
def _get_taxonomy_options() -> Dict[str, List[str]]:
    """Pobiera unikalne wartości pól taksonomii z Qdrant."""
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    client = get_qdrant()
    result = {
        "term_decision_type": [],
        "term_violation_type": [],
        "term_legal_basis": [],
        "term_corrective_measure": [],
        "term_sector": [],
    }
    try:
        offset = None
        while True:
            pts, next_off = client.scroll(
                collection_name=COLLECTION_NAME, limit=500, offset=offset,
                scroll_filter=Filter(must=[
                    FieldCondition(key="doc_type", match=MatchValue(value="uodo_decision"))
                ]),
                with_payload=list(result.keys()), with_vectors=False,
            )
            for pt in (pts or []):
                pay = pt.payload or {}
                for field in result:
                    for val in (pay.get(field) or []):
                        if val and val not in result[field]:
                            result[field].append(val)
            if not next_off or not pts:
                break
            offset = next_off
        for field in result:
            result[field] = sorted(result[field])
    except Exception:
        pass
    return result


def _get_all_tags() -> List[str]:
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


def _extract_tags_with_llm(query: str, available_tags: List[str]) -> List[str]:
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


def _get_matched_tags(query: str) -> List[str]:
    """Zwraca listę tagów pasujących do zapytania (przez LLM)."""
    available_tags = _get_all_tags()
    return _extract_tags_with_llm(query, available_tags)


def _keyword_search_from_query(query: str, filters: Dict = None) -> List[Dict]:
    """Pyta LLM o tagi pasujące do zapytania, następnie pobiera dokumenty exact match."""
    matched_tags = _get_matched_tags(query)
    all_docs, seen_sigs = [], set()
    for tag in matched_tags:
        for d in keyword_exact_search(tag, filters=filters):
            sig = d.get("signature", "")
            if sig not in seen_sigs:
                all_docs.append(d)
                seen_sigs.add(sig)
    return all_docs


def _doc_key(d: Dict) -> str:
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
    query: str, top_k: int = TOP_K, filters: Dict = None, use_graph: bool = True
):
    """Zwraca (List[Dict], List[str]) — dokumenty i użyte tagi."""
    matched_tags = _get_matched_tags(query)

    seen_keys = set()
    merged = []

    if matched_tags:
        # Tagi jako filtr — semantic search zwraca top_k najlepiej pasujących
        for tag in matched_tags:
            tag_filters = (filters or {}).copy()
            tag_filters["keyword"] = tag
            for d in semantic_search(query, top_k=top_k, filters=tag_filters):
                key = _doc_key(d)
                if key not in seen_keys:
                    merged.append(d)
                    seen_keys.add(key)
        merged.sort(key=lambda d: -d.get("_score", 0))

    # Semantic search bez filtra tagów — wszystkie typy razem, ranking po score
    for d in semantic_search(query, top_k=top_k * 2, filters=filters):
        key = _doc_key(d)
        if key not in seen_keys:
            merged.append(d)
            seen_keys.add(key)

    merged.sort(key=lambda d: -d.get("_score", 0))

    if not use_graph or not merged:
        return merged, matched_tags

    # Graf rozszerza tylko orzeczenia UODO
    seed_sigs = [
        d.get("signature", "")
        for d in merged
        if d.get("doc_type") == "uodo_decision" and d.get("signature")
    ]
    if seed_sigs:
        expanded = graph_expand(seed_sigs)
        seen_graph = {
            d.get("signature", "")
            for d in merged
            if d.get("doc_type") == "uodo_decision"
        }
        for sig, rel_type, score in expanded:
            if sig in seen_graph:
                continue
            doc = fetch_by_signature(sig)
            if doc:
                doc["_score"] = score
                doc["_graph_relation"] = rel_type
                merged.append(doc)
                seen_graph.add(sig)

    return merged, matched_tags


# ─────────────────────────── LLM ─────────────────────────────────


def _extract_fragment(content: str, query: str, max_len: int = 1200) -> str:
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
    step = 300
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


def build_context(docs: List[Dict], query: str, max_chars: int = 14000,
                  filters: Dict = None) -> str:
    # Opisz aktywne filtry słownie
    filter_lines = []
    f = filters or {}
    if f.get("status"):
        filter_lines.append(f"Status decyzji: {f['status']}")
    if f.get("term_decision_type"):
        filter_lines.append(f"Rodzaj decyzji: {', '.join(f['term_decision_type'])}")
    if f.get("term_violation_type"):
        filter_lines.append(f"Rodzaj naruszenia: {', '.join(f['term_violation_type'])}")
    if f.get("term_legal_basis"):
        filter_lines.append(f"Podstawa prawna: {', '.join(f['term_legal_basis'])}")
    if f.get("term_corrective_measure"):
        filter_lines.append(f"Środek naprawczy: {', '.join(f['term_corrective_measure'])}")
    if f.get("term_sector"):
        filter_lines.append(f"Sektor: {', '.join(f['term_sector'])}")
    if f.get("keyword"):
        filter_lines.append(f"Słowo kluczowe: {f['keyword']}")

    filter_note = ""
    if filter_lines:
        filter_note = (
            "UWAGA: Wyniki zostały zawężone przez użytkownika za pomocą filtrów:\n"
            + "\n".join(f"  • {l}" for l in filter_lines)
            + "\nOdpowiadaj z uwzględnieniem tego kontekstu filtrowania.\n"
        )

    parts = [
        f"Poniżej znajdują się dokumenty powiązane z pytaniem: «{query}»\n"
        f"Zbiór zawiera DECYZJE UODO oraz ARTYKUŁY ustawy o ochronie danych osobowych.\n"
        f"{filter_note}"
        f"Odpowiadaj WYŁĄCZNIE na podstawie poniższych dokumentów. "
        f"Podawaj sygnatury decyzji [DKN.XXXX] i numery artykułów [Art. X u.o.d.o.].\n"
    ]
    chars = 0
    for i, doc in enumerate(docs, 1):
        dtype = doc.get("doc_type", "")
        graph_rel = doc.get("_graph_relation", "")

        if dtype == "legal_act_article":
            art_num = doc.get("article_num", "?")
            chunk_idx = doc.get("chunk_index", 0)
            total = doc.get("chunk_total", 1)
            text = doc.get("content_text", "")
            label = f"Art. {art_num}"
            if total > 1:
                label += f" (część {chunk_idx + 1}/{total})"
            block = (
                f"[{i}] USTAWA o ochronie danych osobowych — {label}\n"
                f"Źródło: Dz.U. 2019 poz. 1781 (u.o.d.o.)\n"
                f"Treść:\n{text}\n"
            )
        elif dtype in ("gdpr_article", "gdpr_recital"):
            art_num = doc.get("article_num", "?")
            text = doc.get("content_text", "")
            prefix = "Motyw" if dtype == "gdpr_recital" else f"Art. {art_num}"
            block = (
                f"[{i}] RODO (rozporządzenie 2016/679) — {prefix}\n"
                f"Źródło: Dz.Urz. UE L 119/1\n"
                f"Treść:\n{text}\n"
            )
        else:
            sig = doc.get("signature", "?")
            date = doc.get("date_issued", "")[:7]
            status = doc.get("status", "")
            keywords = doc.get("keywords_text", "") or ", ".join(
                doc.get("keywords", [])
            )
            fragment = _extract_fragment(doc.get("content_text", ""), query)
            acts = doc.get("related_acts", [])[:4]
            eu = doc.get("related_eu_acts", [])[:2]

            block = f"[{i}] DECYZJA UODO {sig} ({date}, {status})"
            if graph_rel:
                block += f" [powiązana: {graph_rel}]"
            block += "\n"
            if keywords:
                block += f"Tagi: {keywords[:200]}\n"
            if acts or eu:
                block += f"Akty: {', '.join((acts + eu)[:5])}\n"
            block += f"Treść:\n{fragment}\n"

        if chars + len(block) > max_chars:
            parts.append(f"\n[pominięto {len(docs) - i + 1} dalszych wyników]")
            break
        parts.append(block)
        chars += len(block)

    return "\n---\n".join(parts)


@st.cache_data(ttl=300, show_spinner=False)
def get_available_models(provider: str, api_key: str = None) -> List[str]:
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
    provider: str = None,
    model: str = None,
    api_key: str = None,
):
    """Stream odpowiedzi z Groq lub Ollama Cloud."""
    system = (
        "Jesteś ekspertem ds. ochrony danych osobowych i prawa RODO. "
        "Pomagasz analizować decyzje Prezesa UODO oraz przepisy ustawy o ochronie danych osobowych. "
        "Odpowiadaj po polsku, precyzyjnie i zwięźle. "
        "Zawsze powołuj się na konkretne decyzje UODO podając sygnatury [DKN.XXXX.XX.XXXX] "
        "lub artykuły ustawy [Art. X u.o.d.o.]. "
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
        for chunk in client.chat.completions.create(
            model=model,
            messages=messages,
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


# ─────────────────────────── STATYSTYKI ──────────────────────────


@st.cache_data(ttl=3600)
def get_collection_stats() -> Dict:
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


def decision_url(doc: Dict) -> str:
    sig = doc.get("signature", "")
    url = doc.get("source_url", "")
    if url:
        return url
    slug = sig.lower().replace(".", "_")
    year_m = re.search(r"\b(20\d{2})\b", sig)
    year = year_m.group(1) if year_m else "2024"
    return f"{UODO_PORTAL_BASE}/urn:ndoc:gov:pl:uodo:{year}:{slug}/content"


def render_act_article_card(doc: Dict, rank: int):
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
        <p>{text}{'…' if len(doc.get('content_text',''))>600 else ''}</p>
      </main>
      <footer>
        <small class="text-muted">score: {score:.3f}</small>
      </footer>
    </article>"""
    st.markdown(html, unsafe_allow_html=True)


def render_decision_card(doc: Dict, rank: int):
    """Karta dla decyzji UODO."""
    sig       = doc.get("signature", "?")
    status    = doc.get("status", "")
    date      = doc.get("date_published", "") or doc.get("date_issued", "")
    score     = doc.get("_score", 0)
    source    = doc.get("_source", "")
    graph_rel = doc.get("_graph_relation", "")
    title     = doc.get("title_full", "") or doc.get("title", "")
    name      = doc.get("title", sig)
    url       = decision_url(doc)

    kw_list = doc.get("keywords", [])
    if isinstance(kw_list, str):
        kw_list = [k.strip() for k in kw_list.split(",") if k.strip()]

    kinds   = doc.get("term_decision_type", [])
    sectors = doc.get("term_sector", [])

    # Usuń z keywords to, co już wyświetlamy jako kinds/sectors (unikamy duplikatów)
    taxonomy_values = {v.lower() for v in kinds + sectors}
    kw_list = [k for k in kw_list if k.lower() not in taxonomy_values]

    status_cls = {
        "prawomocna":    "status-final",
        "nieprawomocna": "status-nonfinal",
        "uchylona":      "status-repealed",
    }.get(status, "status-unknown")

    date_fmt = ""
    if date:
        try:
            from datetime import datetime
            d = datetime.strptime(date[:10], "%Y-%m-%d")
            months = ["stycznia","lutego","marca","kwietnia","maja","czerwca",
                      "lipca","sierpnia","września","października","listopada","grudnia"]
            date_fmt = f"{d.day} {months[d.month-1]} {d.year}"
        except Exception:
            date_fmt = date[:10]

    graph_badge = ""
    if source == "graph":
        graph_badge = f' <span class="status-badge status-unknown">↗ {graph_rel or "graf"}</span>'

    # ── Nagłówek karty (HTML — styl portalu)
    st.markdown(f"""
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
        <p class="text-muted">{title[:280] + '…' if len(title) > 280 else title}</p>
      </main>
    </article>""", unsafe_allow_html=True)

    # ── Footer karty — Streamlit (żeby markdown działał poprawnie)
    with st.container():
        # Słowa kluczowe
        if kw_list:
            shown = kw_list[:8]
            rest  = len(kw_list) - len(shown)
            tags  = " · ".join(f"`{k}`" for k in shown)
            suffix = f" *+{rest} więcej*" if rest > 0 else ""
            st.caption(f"🏷️ {tags}{suffix}")

        # Powołane akty
        all_acts = doc.get("related_acts", [])[:4] + doc.get("related_eu_acts", [])[:2]
        if all_acts:
            st.caption("📜 Powołane akty: " + " · ".join(f"`{a}`" for a in all_acts))

        if graph_rel:
            st.caption(f"↗ powiązana przez graf: *{graph_rel}*")

    st.divider()


def render_gdpr_card(doc: Dict, rank: int):
    """Karta dla artykułu lub motywy RODO."""
    art_num       = doc.get("article_num", "?")
    chunk_idx     = doc.get("chunk_index", 0)
    total         = doc.get("chunk_total", 1)
    score         = doc.get("_score", 0)
    text          = doc.get("content_text", "")[:500]
    dtype         = doc.get("doc_type", "")
    chapter       = doc.get("chapter", "")
    chapter_title = doc.get("chapter_title", "")

    is_recital = dtype == "gdpr_recital"
    label = art_num if is_recital else f"Art. {art_num} RODO"
    badge_txt = "motyw RODO" if is_recital else "RODO"
    if not is_recital and total > 1:
        label += f" (część {chunk_idx + 1}/{total})"

    chapter_html = ""
    if chapter and chapter_title:
        chapter_html = f'<small class="text-muted">Rozdział {chapter} — {chapter_title}</small>'

    html = f"""
    <article class="doc-list-item">
      <header>
        <a href="{GDPR_URL}" target="_blank">{label}</a>
        <span class="status-badge status-final">{badge_txt}</span>
      </header>
      <main>
        <h2>{chapter_html}</h2>
        <p>{text}{'…' if len(doc.get("content_text",""))>500 else ''}</p>
      </main>
      <footer>
        <small class="text-muted">score: {score:.3f}</small>
      </footer>
    </article>"""
    st.markdown(html, unsafe_allow_html=True)


def render_card(doc: Dict, rank: int):
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

    st.markdown("""
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
    """, unsafe_allow_html=True)

    # ── Nagłówek portalu ────────────────────────────────────────
    st.markdown("""
    <div class="page-header">
      <div class="page-header-inner">
        <div>
          <h1>Portal Orzeczeń UODO</h1>
          <div class="page-header-sub">Wyszukiwarka decyzji Prezesa UODO · Ustawa o ochronie danych osobowych · RODO</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar — opcje techniczne ──────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Opcje")

        provider = st.selectbox("Provider LLM", ["Ollama Cloud", "Groq"],
                                key="provider_select")
        api_key = st.text_input("Klucz API", type="password",
                                value=st.session_state.get("llm_api_key", ""),
                                key="api_key_input")

        models = get_available_models(provider, api_key)

        default_model = DEFAULT_OLLAMA_MODEL if provider == "Ollama Cloud" else DEFAULT_GROQ_MODEL
        default_idx = next(
            (i for i, m in enumerate(models) if default_model in m), 0
        )
        selected_model = st.selectbox("Model", models, index=default_idx)

        st.session_state["llm_provider"] = provider
        st.session_state["llm_model"]    = selected_model
        st.session_state["llm_api_key"]  = api_key

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

    # ── Filtry ──────────────────────────────────────────────────
    doc_types = []
    if show_decisions:
        doc_types.append("uodo_decision")
    if show_act:
        doc_types.append("legal_act_article")
    if show_gdpr:
        doc_types.extend(["gdpr_article", "gdpr_recital"])
    if not doc_types:
        doc_types = ["uodo_decision", "legal_act_article", "gdpr_article", "gdpr_recital"]

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
            st.markdown('<div class="filter-label">Sygnatura</div>', unsafe_allow_html=True)
            sig_filter = st.text_input("Sygnatura", placeholder="np. DKN.5110",
                                       label_visibility="collapsed", key="sig_filter")
            st.markdown('<div class="filter-label">Status</div>', unsafe_allow_html=True)
            status_filter = st.selectbox(
                "Status", ["— wszystkie —", "prawomocna", "nieprawomocna", "uchylona"],
                label_visibility="collapsed", key="status_filter",
            )
            st.markdown('<div class="filter-label">Słowa kluczowe</div>', unsafe_allow_html=True)
            kw_filter = st.text_input(
                "Słowo kluczowe", placeholder="np. nałożenie kary",
                label_visibility="collapsed", key="kw_filter",
            )
        with fc2:
            st.markdown('<div class="filter-label">Rodzaj decyzji</div>', unsafe_allow_html=True)
            tax_decision = st.multiselect(
                "Rodzaj decyzji", options=taxonomy.get("term_decision_type", []),
                label_visibility="collapsed", key="tax_decision",
            )
            st.markdown('<div class="filter-label">Środek naprawczy</div>', unsafe_allow_html=True)
            tax_measure = st.multiselect(
                "Środek naprawczy", options=taxonomy.get("term_corrective_measure", []),
                label_visibility="collapsed", key="tax_measure",
            )
            st.markdown('<div class="filter-label">Podstawa prawna</div>', unsafe_allow_html=True)
            tax_legal_basis = st.multiselect(
                "Podstawa prawna", options=taxonomy.get("term_legal_basis", []),
                label_visibility="collapsed", key="tax_legal_basis",
            )
        with fc3:
            st.markdown('<div class="filter-label">Rodzaj naruszenia</div>', unsafe_allow_html=True)
            tax_violation = st.multiselect(
                "Rodzaj naruszenia", options=taxonomy.get("term_violation_type", []),
                label_visibility="collapsed", key="tax_violation",
            )
            st.markdown('<div class="filter-label">Sektor</div>', unsafe_allow_html=True)
            tax_sector = st.multiselect(
                "Sektor", options=taxonomy.get("term_sector", []),
                label_visibility="collapsed", key="tax_sector",
            )
            st.markdown('<div class="filter-label">Data ogłoszenia (od–do)</div>',
                        unsafe_allow_html=True)
            dcol1, dcol2 = st.columns(2)
            with dcol1:
                date_from = st.text_input("Od", placeholder="2020-01-01",
                                          label_visibility="collapsed", key="date_from")
            with dcol2:
                date_to = st.text_input("Do", placeholder="2026-12-31",
                                        label_visibility="collapsed", key="date_to")

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
            if col.button(f"{emoji} {question}", key=f"example_{idx}",
                          use_container_width=True):
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

    # Filtr słowa kluczowego i sygnatury — dla wszystkich typów
    if kw_filter.strip():
        filters["keyword"] = kw_filter.strip()

    # ── Wyszukiwanie ────────────────────────────────────────────
    # Dopasuj zapytanie do sygnatury jeśli podano w osobnym polu
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

        with st.spinner("🔍 Wyszukuję..."):
            t0 = time.time()
            _tags = []
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
                        effective_query, top_k=TOP_K, filters=filters, use_graph=use_graph
                    )
            else:
                docs, _tags = hybrid_search(
                    effective_query, top_k=TOP_K, filters=filters, use_graph=use_graph
                )
            search_time = time.time() - t0

        if not docs:
            st.warning("Nie znaleziono dokumentów. Spróbuj zmienić filtry lub sformułowanie.")
            return

        decisions  = [d for d in docs if d.get("doc_type") == "uodo_decision"]
        act_arts   = [d for d in docs if d.get("doc_type") == "legal_act_article"]
        gdpr_docs  = [d for d in docs if d.get("doc_type") in ("gdpr_article", "gdpr_recital")]
        graph_docs = [d for d in docs if d.get("_source") == "graph"]

        _tag_info = f" · tag: `{kw_filter}`" if kw_filter.strip() else ""
        st.caption(
            f"Znaleziono {len(docs)} dokumentów "
            f"({len(decisions)} decyzji, {len(act_arts)} u.o.d.o., "
            f"{len(gdpr_docs)} RODO, {len(graph_docs)} przez graf) · {search_time:.2f}s"
            + _tag_info
        )
        if _tags:
            st.caption(
                "🏷️ Tagi użyte do wyszukiwania: " + " · ".join(f"`{t}`" for t in _tags)
            )

        if use_llm:
            context = build_context(docs, effective_query, filters=filters)
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

        st.markdown(f"### 📋 Dokumenty ({len(docs)})")
        tabs = st.tabs([
            f"Wszystkie ({len(docs)})",
            f"Decyzje UODO ({len(decisions)})",
            f"Ustawa u.o.d.o. ({len(act_arts)})",
            f"RODO ({len(gdpr_docs)})",
            f"Graf ({len(graph_docs)})",
        ])

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
