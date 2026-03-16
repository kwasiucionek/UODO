"""
Wyszukiwanie — Qdrant (semantic + keyword), graf powiązań, tagi LLM, taksonomia.
"""

import os
import pickle
import re
from typing import Any

import networkx as nx
import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue, Range

from config import (
    COLLECTION_NAME,
    DEFAULT_PROVIDER,
    EMBED_MODEL,
    GRAPH_DEPTH,
    GRAPH_PATH,
    GROQ_API_KEY,
    MAX_ACT_DOCS,
    MAX_GDPR_DOCS,
    OLLAMA_CLOUD_API_KEY,
    OLLAMA_CLOUD_URL,
    QDRANT_URL,
    QUERY_STOPWORDS,
    TAXONOMY_STATIC,
    TOP_K,
)


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
                "signature", "doc_type", "related_uodo_rulings",
                "related_acts", "related_eu_acts", "related_court_rulings",
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


# ─────────────────────────── WYSZUKIWANIE QDRANT ─────────────────

def embed(text: str) -> list[float]:
    return get_embedder().encode(text, normalize_embeddings=True).tolist()


def _build_qdrant_filter(filters: dict[str, Any] | None) -> Filter | None:
    """Buduje obiekt Filter z Qdrant na podstawie słownika filtrów."""
    must = []
    if not filters:
        return None
    if filters.get("status"):
        must.append(FieldCondition(key="status", match=MatchValue(value=filters["status"])))
    if filters.get("keyword"):
        must.append(FieldCondition(key="keywords", match=MatchValue(value=filters["keyword"])))
    if filters.get("doc_types"):
        must.append(FieldCondition(key="doc_type", match=MatchAny(any=filters["doc_types"])))
    if filters.get("year_from") or filters.get("year_to"):
        must.append(FieldCondition(
            key="year",
            range=Range(gte=filters.get("year_from", 2000), lte=filters.get("year_to", 2030)),
        ))
    for term_field in (
        "term_decision_type", "term_violation_type",
        "term_legal_basis", "term_corrective_measure", "term_sector",
    ):
        vals = filters.get(term_field, [])
        if vals:
            must.append(FieldCondition(key=term_field, match=MatchAny(any=vals)))
    return Filter(must=must) if must else None


def semantic_search(
    query: str,
    top_k: int = TOP_K,
    filters: dict[str, Any] | None = None,
    score_threshold: float = 0.25,
) -> list[dict[str, Any]]:
    vec = get_qdrant()
    client = get_qdrant()
    res = client.query_points(
        collection_name=COLLECTION_NAME,
        query=embed(query),
        limit=top_k,
        query_filter=_build_qdrant_filter(filters),
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


def keyword_exact_search(
    keyword: str, filters: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Pobiera WSZYSTKIE dokumenty z danym tagiem (scroll z paginacją, bez limitu)."""
    client = get_qdrant()
    kw_filters = {**(filters or {}), "keyword": keyword}
    qdrant_filter = _build_qdrant_filter(kw_filters)
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


def fetch_by_signature(sig: str) -> dict[str, Any] | None:
    client = get_qdrant()
    pts, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(must=[
            FieldCondition(key="signature", match=MatchValue(value=sig)),
            FieldCondition(key="doc_type", match=MatchValue(value="uodo_decision")),
        ]),
        limit=1,
        with_payload=True,
    )
    if pts:
        d = (pts[0].payload or {}).copy()
        d["_source"] = "graph"
        d["_score"] = 0.0
        return d
    return None


# ─────────────────────────── GRAF POWIĄZAŃ ───────────────────────

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
        decay = 0.65 ** d
        new_frontier: set[str] = set()
        for node in frontier:
            if node not in G:
                continue
            for nb in G.successors(node):
                if nb not in visited and G[node][nb].get("relation") == "CITES_UODO":
                    result.append((nb, "cytowana", 0.6 * decay))
                    visited.add(nb)
                    new_frontier.add(nb)
            for nb in G.predecessors(node):
                if nb not in visited and (
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


# ─────────────────────────── TAGI I TAKSONOMIA ───────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def get_taxonomy_options() -> dict[str, list[str]]:
    """Opcje filtrów taksonomii. Statyczne wartości + dynamiczne z Qdrant."""
    result = {k: list(v) for k, v in TAXONOMY_STATIC.items()}
    dynamic_fields = [f for f, v in TAXONOMY_STATIC.items() if not v]
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
                scroll_filter=Filter(must=[
                    FieldCondition(key="doc_type", match=MatchValue(value="uodo_decision"))
                ]),
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
def get_all_tags() -> list[str]:
    """Wszystkie unikalne tagi z kolekcji (cache 5 min)."""
    client = get_qdrant()
    all_tags: set[str] = set()
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


def extract_tags_with_llm(query: str, available_tags: list[str]) -> list[str]:
    """Pyta LLM o tagi pasujące do zapytania (fallback gdy brak bezpośredniego trafienia)."""
    provider = st.session_state.get("llm_provider", DEFAULT_PROVIDER)
    api_key  = st.session_state.get("llm_api_key", "")
    model    = st.session_state.get("llm_model", "")

    tags_list = "\n".join(f"- {t}" for t in available_tags)
    prompt = (
        f"Masz listę tagów z bazy orzeczeń UODO (organ ochrony danych osobowych w Polsce).\n"
        f"Wybierz tagi NAJBARDZIEJ pasujące do zapytania — maksymalnie 8 tagów z listy.\n"
        f"Jeśli temat zapytania nie jest pokryty przez żaden istniejący tag, możesz dodać maksymalnie 4 NOWE tagi spoza listy.\n"
        f"Uwzględnij synonimy i formy fleksyjne (np. 'kampania wyborcza' → szukaj tagów o wyborach, partiach, polityce).\n"
        f"Wybieraj tylko tagi ŚCIŚLE związane z tematem — nie wybieraj zbyt ogólnych tagów.\n"
        f"Odpowiedz TYLKO listą tagów, jeden na linię, bez komentarzy.\n"
        f"Tagi z listy — dokładna pisownia. Nowe tagi — z prefiksem [NOWY].\n"
        f"Zapytanie: {query}\n\nDostępne tagi:\n{tags_list}"
    )

    try:
        if provider == "Groq":
            from groq import Groq
            client = Groq(api_key=api_key or GROQ_API_KEY)
            resp = client.chat.completions.create(
                model=model, max_tokens=400, stream=False,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.choices[0].message.content or ""
        else:
            import requests as _req
            resp = _req.post(
                f"{OLLAMA_CLOUD_URL}/api/chat",
                headers={"Authorization": f"Bearer {api_key or OLLAMA_CLOUD_API_KEY}"},
                json={"model": model, "stream": False, "messages": [{"role": "user", "content": prompt}]},
                timeout=30,
            )
            raw = resp.json().get("message", {}).get("content", "")

        tags_lower = {t.lower(): t for t in available_tags}
        existing_found, new_found = [], []
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


def get_matched_tags(query: str) -> list[str]:
    """Zwraca tagi pasujące do zapytania (przez LLM) — wywoływane jako fallback."""
    return extract_tags_with_llm(query, get_all_tags())


# ─────────────────────────── DEDUPLIKACJA ────────────────────────

def doc_key(d: dict[str, Any]) -> str:
    """Unikalny klucz dokumentu do deduplikacji."""
    doc_id = d.get("doc_id", "")
    if doc_id:
        return doc_id
    sig   = d.get("signature", "")
    dtype = d.get("doc_type", "")
    art   = d.get("article_num", "")
    chunk = d.get("chunk_index", 0)
    if dtype in ("legal_act_article", "gdpr_article", "gdpr_recital"):
        return f"{dtype}:{sig}:{art}:{chunk}"
    return sig or f"{dtype}:{art}"


# ─────────────────────────── HYBRID SEARCH ───────────────────────

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
    # Tagi LLM — wywołujemy raz, używamy tylko jako fallback
    matched_tags = get_matched_tags(query)

    seen_keys: set[str] = set()
    decisions: list[dict[str, Any]] = []
    act_docs:  list[dict[str, Any]] = []
    gdpr_docs: list[dict[str, Any]] = []

    def _add(bucket: list, doc: dict) -> bool:
        key = doc_key(doc)
        if key in seen_keys:
            return False
        seen_keys.add(key)
        bucket.append(doc)
        return True

    # Filtry bez pola "keyword" — keyword stosujemy osobno per wyszukiwanie
    filters_base = {k: v for k, v in (filters or {}).items() if k != "keyword"}

    # ═══════════════════════════════════════════════════════════════
    # DECYZJE UODO — pełny recall
    # ═══════════════════════════════════════════════════════════════

    # 1a. Explicit keyword z UI (filtr ręczny użytkownika)
    explicit_keyword = (filters or {}).get("keyword", "")
    if explicit_keyword:
        for d in keyword_exact_search(explicit_keyword, {**filters_base, "doc_types": ["uodo_decision"]}):
            _add(decisions, d)

    # 1b. Frazy bezpośrednio z zapytania → scroll po tagu BEZ LLM
    # Szukamy jednwyrazowych i dwuwyrazowych fraz z query w liście tagów.
    # Unikamy przez to "dane biometryczne" gdy user pyta o "dane genetyczne".
    words = [
        w.lower() for w in re.split(r"\W+", query)
        if w.lower() not in QUERY_STOPWORDS and len(w) > 2
    ]
    direct_phrases: list[str] = list(dict.fromkeys(
        words + [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
    ))
    all_tags_lower = {t.lower(): t for t in get_all_tags()}
    direct_hits = [all_tags_lower[p] for p in direct_phrases if p in all_tags_lower]

    for tag in direct_hits:
        for d in keyword_exact_search(tag, {**filters_base, "doc_types": ["uodo_decision"]}):
            _add(decisions, d)

    # 1c. Tagi LLM — tylko gdy 1b nic nie znalazło (fraza nie jest tagiem)
    if not decisions and matched_tags:
        for tag in matched_tags:
            for d in keyword_exact_search(tag, {**filters_base, "doc_types": ["uodo_decision"]}):
                _add(decisions, d)

    # 1d. Semantic — ostatni fallback gdy tagi nic nie dały
    if len(decisions) < 5:
        for d in semantic_search(query, top_k=20, filters={**filters_base, "doc_types": ["uodo_decision"]}, score_threshold=0.45):
            _add(decisions, d)

    decisions.sort(key=lambda d: -d.get("_score", 0))

    # ═══════════════════════════════════════════════════════════════
    # ARTYKUŁY u.o.d.o. — pomocnicze, max MAX_ACT_DOCS
    # ═══════════════════════════════════════════════════════════════

    if explicit_keyword:
        for d in keyword_exact_search(explicit_keyword, {**filters_base, "doc_types": ["legal_act_article"]}):
            if len(act_docs) >= MAX_ACT_DOCS:
                break
            _add(act_docs, d)

    if len(act_docs) < MAX_ACT_DOCS:
        for d in semantic_search(query, top_k=MAX_ACT_DOCS - len(act_docs),
                                  filters={**filters_base, "doc_types": ["legal_act_article"]},
                                  score_threshold=0.25):
            if len(act_docs) >= MAX_ACT_DOCS:
                break
            _add(act_docs, d)

    # ═══════════════════════════════════════════════════════════════
    # ARTYKUŁY RODO — kontekst prawny, max MAX_GDPR_DOCS
    # ═══════════════════════════════════════════════════════════════

    gdpr_types = ["gdpr_article", "gdpr_recital"]

    if explicit_keyword:
        for d in keyword_exact_search(explicit_keyword, {**filters_base, "doc_types": gdpr_types}):
            if len(gdpr_docs) >= MAX_GDPR_DOCS:
                break
            _add(gdpr_docs, d)

    if len(gdpr_docs) < MAX_GDPR_DOCS:
        for d in semantic_search(query, top_k=MAX_GDPR_DOCS - len(gdpr_docs),
                                  filters={**filters_base, "doc_types": gdpr_types},
                                  score_threshold=0.3):
            if len(gdpr_docs) >= MAX_GDPR_DOCS:
                break
            _add(gdpr_docs, d)

    # ═══════════════════════════════════════════════════════════════
    # Złącz: decyzje pierwsze, u.o.d.o., RODO
    # ═══════════════════════════════════════════════════════════════
    merged = decisions + act_docs + gdpr_docs

    if not use_graph or not decisions:
        return merged, matched_tags

    # Graf — rozszerza tylko decyzje UODO
    seed_sigs = [d.get("signature", "") for d in decisions if d.get("signature")]
    if seed_sigs:
        seen_graph = {d.get("signature", "") for d in decisions}
        for sig, rel_type, score in graph_expand(seed_sigs):
            if sig in seen_graph:
                continue
            doc = fetch_by_signature(sig)
            if doc:
                doc["_score"] = score
                doc["_graph_relation"] = rel_type
                decisions.append(doc)
                seen_graph.add(sig)
        merged = decisions + act_docs + gdpr_docs

    return merged, matched_tags


# ─────────────────────────── STATYSTYKI ──────────────────────────

@st.cache_data(ttl=3600)
def get_collection_stats() -> dict[str, Any]:
    client = get_qdrant()
    info  = client.get_collection(COLLECTION_NAME)
    total = info.points_count

    decision_count = 0
    act_chunk_count = 0
    offset = None
    while True:
        pts, next_off = client.scroll(
            collection_name=COLLECTION_NAME, limit=500, offset=offset,
            with_payload=["doc_type"], with_vectors=False,
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
    graph_stats: dict[str, Any] = {}
    if G:
        uodo = [n for n, d in G.nodes(data=True) if d.get("doc_type") == "uodo_decision"]
        most_cited = sorted(
            [(n, G.in_degree(n)) for n in uodo if G.in_degree(n) > 0],
            key=lambda x: -x[1],
        )[:5]
        graph_stats = {"edges": G.number_of_edges(), "most_cited": most_cited}

    return {"total": total, "decisions": decision_count, "act_chunks": act_chunk_count, **graph_stats}
