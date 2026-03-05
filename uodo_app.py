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
_RE_QUERY_SIG = re.compile(
    r'^\s*([A-Z]{2,6}\.\d{3,5}\.\d+\.\d{4})\s*$', re.IGNORECASE
)

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

GROQ_API_KEY         = os.getenv("GROQ_API_KEY", "")
OLLAMA_CLOUD_API_KEY = os.getenv("OLLAMA_CLOUD_API_KEY", "")
OLLAMA_CLOUD_URL     = os.getenv("OLLAMA_CLOUD_URL", "https://ollama.com")
OLLAMA_LOCAL_URL     = os.getenv("OLLAMA_LOCAL_URL", "http://localhost:11434")

PROVIDERS = ["Ollama Cloud", "Groq"]
DEFAULT_PROVIDER = "Ollama Cloud"
DEFAULT_OLLAMA_MODEL = "gpt-oss:120b"
DEFAULT_GROQ_MODEL   = "openai/gpt-oss-120b"

TOP_K = 8
GRAPH_DEPTH = 2
UODO_PORTAL_BASE = "https://orzeczenia.uodo.gov.pl/document"
ISAP_ACT_URL  = "https://isap.sejm.gov.pl/isap.nsf/DocDetails.xsp?id=WDU20190001781"
GDPR_URL      = "https://eur-lex.europa.eu/legal-content/PL/TXT/?uri=CELEX:32016R0679"


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
            collection_name=COLLECTION_NAME, limit=500, offset=offset,
            with_payload=["signature", "doc_type", "related_uodo_rulings",
                          "related_acts", "related_eu_acts", "related_court_rulings"],
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
            must.append(FieldCondition(key="status", match=MatchValue(value=filters["status"])))

        if filters.get("keyword"):
            must.append(FieldCondition(key="keywords", match=MatchValue(value=filters["keyword"])))
        if filters.get("doc_types"):
            must.append(FieldCondition(
                key="doc_type",
                match=MatchAny(any=filters["doc_types"])
            ))

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
    for hit in (res.points or []):
        d = (hit.payload or {}).copy()
        d["_score"] = hit.score
        d["_source"] = "semantic"
        docs.append(d)
    return docs


def graph_expand(seed_sigs: List[str], depth: int = GRAPH_DEPTH) -> List[Tuple[str, str, float]]:
    G = get_graph()
    if G is None:
        return []

    visited = set(seed_sigs)
    result = []
    frontier = set(seed_sigs)

    for d in range(depth):
        decay = 0.65 ** d
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
                if (G[nb][node].get("relation") == "CITES_UODO"
                        and G.nodes.get(nb, {}).get("doc_type") == "uodo_decision"):
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
        scroll_filter=Filter(must=[
            FieldCondition(key="signature", match=MatchValue(value=sig)),
            FieldCondition(key="doc_type", match=MatchValue(value="uodo_decision")),
        ]),
        limit=1, with_payload=True,
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
            must.append(FieldCondition(key="status", match=MatchValue(value=filters["status"])))
        if filters.get("year_from") or filters.get("year_to"):
            must.append(FieldCondition(key="year", range=Range(
                gte=filters.get("year_from", 2000),
                lte=filters.get("year_to", 2030),
            )))
        if filters.get("doc_types"):
            must.append(FieldCondition(key="doc_type", match=MatchAny(any=filters["doc_types"])))

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
        for pt in (pts or []):
            d = (pt.payload or {}).copy()
            d["_score"] = 1.0
            d["_source"] = "keyword"
            docs.append(d)
        if next_offset is None or not pts:
            break
        offset = next_offset
    return docs


@st.cache_data(ttl=300, show_spinner=False)
def _get_all_tags() -> List[str]:
    """Pobiera wszystkie unikalne tagi z kolekcji (cache 5 min)."""
    client = get_qdrant()
    all_tags = set()
    offset = None
    while True:
        pts, next_offset = client.scroll(
            collection_name=COLLECTION_NAME, limit=500,
            with_payload=["keywords"], with_vectors=False,
            offset=offset,
        )
        for pt in (pts or []):
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
    """Pyta LLM o tagi pasujące do zapytania. Zwraca listę tagów."""
    provider = st.session_state.get("llm_provider", DEFAULT_PROVIDER)
    api_key  = st.session_state.get("llm_api_key", "")
    model    = st.session_state.get("llm_model", "")

    tags_list = "\n".join(f"- {t}" for t in available_tags)
    prompt = (
        f"Masz listę tagów z bazy orzeczeń UODO (organ ochrony danych osobowych w Polsce).\n"
        f"Wybierz tagi NAJBARDZIEJ pasujące do zapytania — maksymalnie 5 tagów.\n"
        f"Uwzględnij synonimy i formy fleksyjne (np. 'kampania wyborcza' → szukaj tagów o wyborach, partiach, polityce).\n"
        f"Wybieraj tylko tagi ŚCIŚLE związane z tematem — nie wybieraj ogólnych tagów jak 'nałożenie kary' czy 'podstawy prawne' chyba że zapytanie wprost o nie pyta.\n"
        f"Odpowiedz TYLKO listą tagów z poniższej listy, jeden na linię, bez komentarzy.\n"
        f"Jeśli żaden tag nie pasuje — odpowiedz pustą linią.\n\n"
        f"Zapytanie: {query}\n\n"
        f"Dostępne tagi:\n{tags_list}"
    )

    try:
        if provider == "Groq":
            from groq import Groq
            client = Groq(api_key=api_key or GROQ_API_KEY)
            resp = client.chat.completions.create(
                model=model, max_tokens=300, stream=False,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.choices[0].message.content or ""
        else:
            import requests as _req
            resp = _req.post(
                f"{OLLAMA_CLOUD_URL}/api/chat",
                headers={"Authorization": f"Bearer {api_key or OLLAMA_CLOUD_API_KEY}"},
                json={"model": model, "stream": False,
                      "messages": [{"role": "user", "content": prompt}]},
                timeout=30,
            )
            raw = resp.json().get("message", {}).get("content", "")

        found = []
        tags_lower = {t.lower(): t for t in available_tags}
        for line in raw.strip().splitlines():
            line = line.strip().lstrip("- ").strip()
            if line.lower() in tags_lower:
                found.append(tags_lower[line.lower()])
        return found
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


def hybrid_search(query: str, top_k: int = TOP_K,
                  filters: Dict = None, use_graph: bool = True):
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
        d.get("signature", "") for d in merged
        if d.get("doc_type") == "uodo_decision" and d.get("signature")
    ]
    if seed_sigs:
        expanded = graph_expand(seed_sigs)
        seen_graph = {d.get("signature", "") for d in merged if d.get("doc_type") == "uodo_decision"}
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
    stopwords = {"jakie", "są", "w", "o", "i", "z", "do", "na", "co", "ile",
                 "jak", "czy", "przez", "dla", "po", "przy", "od", "ze", "to"}
    keywords = [w.lower() for w in re.split(r'\W+', query)
                if w.lower() not in stopwords and len(w) > 2]
    if not keywords:
        return content[:max_len]
    step = 300
    best_score, best_pos = -1, 0
    cl = content.lower()
    for pos in range(0, max(1, len(content) - max_len), step):
        score = sum(cl[pos:pos + max_len].count(kw) for kw in keywords)
        if score > best_score:
            best_score, best_pos = score, pos
    fragment = content[best_pos:best_pos + max_len]
    if best_pos > 0:
        nl = fragment.find("\n")
        if 0 < nl < 150:
            fragment = fragment[nl:].lstrip()
        fragment = "[…]\n" + fragment
    return fragment


def build_context(docs: List[Dict], query: str, max_chars: int = 14000) -> str:
    parts = [
        f"Poniżej znajdują się dokumenty powiązane z pytaniem: «{query}»\n"
        f"Zbiór zawiera DECYZJE UODO oraz ARTYKUŁY ustawy o ochronie danych osobowych.\n"
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
                label += f" (część {chunk_idx+1}/{total})"
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
            keywords = doc.get("keywords_text", "") or ", ".join(doc.get("keywords", []))
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
                m.id for m in models_resp.data
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


def call_llm_stream(query: str, context: str, provider: str = None,
                    model: str = None, api_key: str = None):
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
    messages = [{"role": "system", "content": system},
                {"role": "user",   "content": user}]

    provider = provider or st.session_state.get("llm_provider", DEFAULT_PROVIDER)
    model    = model    or st.session_state.get("llm_model", "")
    api_key  = api_key  or st.session_state.get("llm_api_key", "")

    if provider == "Groq":
        from groq import Groq
        client = Groq(api_key=api_key or GROQ_API_KEY)
        for chunk in client.chat.completions.create(
            model=model, messages=messages,
            max_tokens=2048, stream=True,
        ):
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    elif provider == "Ollama Cloud":
        import requests as _req
        import json as _json
        resp = _req.post(
            f"{OLLAMA_CLOUD_URL}/api/chat",
            headers={"Authorization": f"Bearer {api_key or OLLAMA_CLOUD_API_KEY}"},
            json={"model": model, "messages": messages, "stream": True},
            stream=True, timeout=120,
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
    graph_stats = {}
    if G:
        uodo = [n for n, d in G.nodes(data=True) if d.get("doc_type") == "uodo_decision"]
        most_cited = sorted([(n, G.in_degree(n)) for n in uodo if G.in_degree(n) > 0],
                            key=lambda x: -x[1])[:5]
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
    text = doc.get("content_text", "")

    label = f"Art. {art_num} u.o.d.o."
    if total > 1:
        label += f" (część {chunk_idx + 1}/{total})"

    with st.container():
        col1, col2 = st.columns([5, 1])
        with col1:
            st.markdown(
                f"**{rank}. 📗 [{label}]({ISAP_ACT_URL})** "
                f"— Ustawa o ochronie danych osobowych"
            )
        with col2:
            st.caption("📖 ustawa")
            st.caption(f"score: {score:.3f}")

        st.caption("📅 Dz.U. 2019 poz. 1781 · tekst jednolity z 30 sierpnia 2019 r.")

        with st.expander("📄 Treść artykułu", expanded=True):
            st.markdown(f"<small>{text}</small>", unsafe_allow_html=True)

        st.divider()


def render_decision_card(doc: Dict, rank: int):
    """Karta dla decyzji UODO."""
    sig = doc.get("signature", "?")
    status = doc.get("status", "")
    date = doc.get("date_issued", "")
    score = doc.get("_score", 0)
    source = doc.get("_source", "")
    graph_rel = doc.get("_graph_relation", "")
    kw_list = doc.get("keywords", [])
    if isinstance(kw_list, str):
        kw_list = [k.strip() for k in kw_list.split(",") if k.strip()]

    status_icon = {"prawomocna": "🟢", "nieprawomocna": "🟡"}.get(status, "⚪")
    source_badge = "📊 graf" if source == "graph" else "🔍 semantic"

    with st.container():
        col1, col2 = st.columns([5, 1])
        with col1:
            st.markdown(f"**{rank}. [{sig}]({decision_url(doc)})** {status_icon} {status}")
            if graph_rel:
                st.caption(f"↗ powiązana przez graf: *{graph_rel}*")
        with col2:
            st.caption(source_badge)
            st.caption(f"score: {score:.3f}")

        meta_cols = st.columns(3)
        with meta_cols[0]:
            st.caption(f"📅 {date[:10] if date else '—'}")
        with meta_cols[1]:
            acts = doc.get("related_acts", [])
            if acts:
                st.caption(f"📜 {len(acts)} aktów")
        with meta_cols[2]:
            courts = doc.get("related_court_rulings", [])
            if courts:
                st.caption(f"⚖️ {len(courts)} wyroków")

        if kw_list:
            tags = " · ".join(f"`{k}`" for k in kw_list[:8])
            suffix = f" *+{len(kw_list) - 8} więcej*" if len(kw_list) > 8 else ""
            st.caption(f"🏷️ {tags}{suffix}")

        content = doc.get("content_text", "")
        if content:
            with st.expander("📄 Fragment treści", expanded=False):
                st.markdown(f"<small>{content[:800]}…</small>", unsafe_allow_html=True)

        all_acts = doc.get("related_acts", [])[:4] + doc.get("related_eu_acts", [])[:2]
        if all_acts:
            st.caption("Powołane akty: " + " · ".join(f"`{a}`" for a in all_acts))

        st.divider()


def render_gdpr_card(doc: Dict, rank: int):
    """Karta dla artykułu lub motywy RODO."""
    art_num = doc.get("article_num", "?")
    chunk_idx = doc.get("chunk_index", 0)
    total = doc.get("chunk_total", 1)
    score = doc.get("_score", 0)
    text = doc.get("content_text", "")
    dtype = doc.get("doc_type", "")
    chapter = doc.get("chapter", "")
    chapter_title = doc.get("chapter_title", "")

    is_recital = dtype == "gdpr_recital"
    icon = "💬" if is_recital else "🇪🇺"
    label = f"{icon} {art_num} RODO"
    if not is_recital and total > 1:
        label += f" (część {chunk_idx + 1}/{total})"

    with st.container():
        col1, col2 = st.columns([5, 1])
        with col1:
            st.markdown(f"**{rank}. [{label}]({GDPR_URL})**")
            if chapter and chapter_title:
                st.caption(f"Rozdział {chapter} — {chapter_title}")
        with col2:
            st.caption("🇪🇺 RODO")
            st.caption(f"score: {score:.3f}")

        with st.expander("📄 Treść", expanded=not is_recital):
            st.markdown(f"<small>{text}</small>", unsafe_allow_html=True)

        st.divider()


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
        page_title="UODO RAG — Wyszukiwarka Decyzji i Przepisów",
        page_icon="🔐",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown("""
    <style>
        .main-header { font-size: 2rem; font-weight: 700; color: #1a365d; margin-bottom: 0; }
        .sub-header { color: #4a5568; font-size: 0.95rem; margin-bottom: 1.5rem; }
        .answer-box { background: #f0f7ff; border-left: 4px solid #3182ce;
                      padding: 1rem 1.2rem; border-radius: 6px; margin: 1rem 0; }
        div[data-testid="stExpander"] { border: 1px solid #e2e8f0; border-radius: 6px; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-header">🔐 UODO RAG</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Wyszukiwarka decyzji Prezesa UODO i przepisów ustawy '
        'o ochronie danych osobowych · Graf powiązań · Analiza AI</div>',
        unsafe_allow_html=True,
    )

    # ── Sidebar ────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Opcje")

        st.markdown("### 🤖 Model AI")
        provider = st.selectbox("Provider", PROVIDERS,
                                index=PROVIDERS.index(DEFAULT_PROVIDER))

        if provider == "Groq":
            _key_env = GROQ_API_KEY
            _key_label = "Groq API Key"
            _key_env_name = "GROQ_API_KEY"
        else:
            _key_env = OLLAMA_CLOUD_API_KEY
            _key_label = "Ollama Cloud API Key"
            _key_env_name = "OLLAMA_CLOUD_API_KEY"

        if _key_env:
            st.caption(f"🔑 Klucz z .env ({_key_env_name})")
            api_key = _key_env
        else:
            api_key = st.text_input(_key_label, type="password")

        models = get_available_models(provider, api_key)
        default_model = DEFAULT_OLLAMA_MODEL if provider == "Ollama Cloud" else DEFAULT_GROQ_MODEL
        default_idx = next(
            (i for i, m in enumerate(models) if default_model in m),
            0
        )
        selected_model = st.selectbox("Model", models, index=default_idx)

        st.session_state["llm_provider"] = provider
        st.session_state["llm_model"]    = selected_model
        st.session_state["llm_api_key"]  = api_key

        st.markdown("---")

        use_graph = st.toggle("Graf powiązań", value=True)
        use_llm = st.toggle("Odpowiedź AI", value=True)

        st.markdown("### 📂 Typ dokumentów")
        show_decisions = st.checkbox("Decyzje UODO", value=True)
        show_act = st.checkbox("Ustawa o ochronie danych (u.o.d.o.)", value=True)
        show_gdpr = st.checkbox("RODO (rozporządzenie UE 2016/679)", value=True)

        st.markdown("### 🔍 Filtry decyzji")
        status_filter = st.selectbox(
            "Status", ["— wszystkie —", "prawomocna", "nieprawomocna"]
        )

        kw_filter = st.text_input(
            "Słowo kluczowe (tag)",
            placeholder="np. Nałożenie kary",
        )

        st.markdown("---")

        try:
            stats = get_collection_stats()
            st.markdown("### 📊 Baza wiedzy")
            st.metric("Decyzje UODO", stats.get("decisions", 0))
            st.metric("Artykuły u.o.d.o.", stats.get("act_chunks", 0))
            if stats.get("edges"):
                st.metric("Powiązania w grafie", stats.get("edges", 0))
            if stats.get("most_cited"):
                st.markdown("**Najczęściej cytowane:**")
                for sig, cnt in stats["most_cited"]:
                    st.caption(f"`{sig}` — {cnt}×")
        except Exception:
            pass

    # ── Filtry ────────────────────────────────────────────────
    doc_types = []
    if show_decisions:
        doc_types.append("uodo_decision")
    if show_act:
        doc_types.append("legal_act_article")
    if show_gdpr:
        doc_types.extend(["gdpr_article", "gdpr_recital"])
    if not doc_types:
        doc_types = ["uodo_decision", "legal_act_article", "gdpr_article", "gdpr_recital"]

    filters = {"doc_types": doc_types}
    if status_filter != "— wszystkie —":
        filters["status"] = status_filter

    if kw_filter.strip():
        filters["keyword"] = kw_filter.strip()

    # ── Pole zapytania ─────────────────────────────────────────
    if "_example_query" in st.session_state:
        st.session_state["query_input"] = st.session_state.pop("_example_query")

    query = st.text_input(
        "Zadaj pytanie lub wpisz temat:",
        placeholder="np. Kiedy administrator musi zgłosić naruszenie danych osobowych?",
        key="query_input",
    )
    search_btn = st.button("🔍 Szukaj", type="primary")

    # ── Przykładowe pytania — zawsze widoczne ─────────────────
    with st.expander("💡 **Przykładowe pytania**", expanded=not bool(query)):
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
        ]
        cols = st.columns(2)
        for idx, (emoji, question) in enumerate(examples):
            col = cols[idx % 2]
            if col.button(f"{emoji} {question}", key=f"example_{idx}",
                           use_container_width=True):
                st.session_state["_example_query"] = question
                st.rerun()

    if query and (search_btn or st.session_state.get("last_query") != query
                  or st.session_state.get("last_filters") != str(filters)):
        st.session_state["last_query"] = query
        st.session_state["last_filters"] = str(filters)

        with st.spinner("🔍 Wyszukuję..."):
            t0 = time.time()
            _tags = []
            sig_match = _RE_QUERY_SIG.match(query)
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
                    st.warning(f"Nie znaleziono decyzji o sygnaturze **{sig_norm}** w bazie.")
                    docs, _tags = hybrid_search(query, top_k=TOP_K, filters=filters, use_graph=use_graph)
            else:
                docs, _tags = hybrid_search(query, top_k=TOP_K, filters=filters, use_graph=use_graph)
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
            f"{len(gdpr_docs)} RODO, {len(graph_docs)} przez graf) · {search_time:.2f}s" + _tag_info
        )
        if _tags:
            st.caption("🏷️ Tagi użyte do wyszukiwania: " + " · ".join(f"`{t}`" for t in _tags))

        if use_llm:
            context = build_context(docs, query)
            st.markdown("### 💬 Odpowiedź AI")
            answer_placeholder = st.empty()
            full_answer = ""
            try:
                for chunk in call_llm_stream(query, context):
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
