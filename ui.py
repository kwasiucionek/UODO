"""
Interfejs użytkownika — karty wyników, budowanie kontekstu LLM, CSS.
"""

import re
from typing import Any

import streamlit as st
from config import GDPR_URL, ISAP_ACT_URL, UODO_PORTAL_BASE
from models import (
    CONTEXT_TYPE_ORDER,
    TPL_ACT_ARTICLE,
    TPL_DECISION,
    TPL_GDPR,
    TPL_HEADER,
    AgentMemory,
)

# ─────────────────────────── CSS PORTALU UODO ────────────────────

UODO_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Red+Hat+Display:wght@400;500;600;700;800&display=swap');

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
        --uodo-border-radius: 2px;
    }

    /* ── Jawne białe tło — nadpisuje dark mode Streamlit ── */
    .stApp,
    .stApp > div,
    section[data-testid="stSidebar"] ~ div {
        background-color: #ffffff !important;
    }
    html, body, [class*="css"] {
        font-family: 'Red Hat Display', sans-serif !important;
        color: var(--body-color);
        background-color: #ffffff !important;
    }
    html, body, [class*="css"] {
        font-family: 'Red Hat Display', sans-serif !important;
        color: var(--body-color);
    }

    [data-testid="stHeader"]  { background: transparent !important; box-shadow: none !important; }
    footer                    { display: none; }
    .main .block-container    { padding-top: 0 !important; max-width: 1150px; }

    [data-testid="stSidebar"] { background: var(--uodo-blue-80); }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] p { color: #c5d3e8 !important; }
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: white !important; }
    [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.15); }

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

    .featured-card {
        background-color: var(--uodo-blue-20);
        padding: 2rem 2.5rem;
        border-radius: var(--uodo-border-radius);
        margin-bottom: 1.5rem;
    }

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
    article.doc-list-item > header time { color: var(--uodo-dark-gray); font-size: 0.85rem; }
    article.doc-list-item:hover > header { background-color: var(--uodo-blue-50); }
    article.doc-list-item:hover > header > a,
    article.doc-list-item:hover > header time,
    article.doc-list-item:hover > header small { color: var(--uodo-white) !important; }
    article.doc-list-item > main { color: var(--uodo-dark-gray); padding: 0 20px; }
    article.doc-list-item > main > * { display: block; margin-bottom: 16px; }
    article.doc-list-item > main > *:first-child { margin-top: 16px; }
    article.doc-list-item > main h2 {
        font-weight: 700; font-size: 1rem; line-height: 150%;
        color: var(--uodo-dark-gray); margin: 0 0 8px;
    }
    article.doc-list-item > main h2 a { color: var(--uodo-dark-gray); text-decoration: none; }
    article.doc-list-item > main h2 a:hover { color: var(--uodo-blue-40); }
    article.doc-list-item > main a { color: var(--uodo-dark-gray); font-size: 0.92rem; text-decoration: none; }
    article.doc-list-item > main p { margin: 0; font-size: 0.92rem; }
    article.doc-list-item > footer {
        margin: 0 20px;
        border-top: 1px solid var(--uodo-blue-30);
        padding: 12px 0 14px;
        overflow: hidden;
    }

    .status-badge {
        display: inline-block; padding: 2px 10px; border-radius: 2px;
        font-size: 0.75rem; font-weight: 600; white-space: nowrap;
    }
    .status-final    { background: #d1fae5; color: #065f46; }
    .status-nonfinal { background: #dbeafe; color: #1e40af; }
    .status-repealed { background: #f3f4f6; color: #374151; }
    .status-unknown  { background: #fef9c3; color: #713f12; }

    .answer-box {
        background: var(--uodo-blue-10);
        border-left: 4px solid var(--uodo-blue-60);
        padding: 1rem 1.2rem;
        border-radius: 2px;
        margin: 0.5rem 0 1rem;
        font-family: 'Red Hat Display', sans-serif;
    }

    a { color: var(--link-color); text-decoration: none; }
    a:hover { color: var(--link-hover-color); }

    [data-testid="stTabs"] [data-baseweb="tab"] {
        font-family: 'Red Hat Display', sans-serif !important;
        font-size: 0.88rem !important;
    }
    [data-testid="stTabs"] [aria-selected="true"] {
        color: var(--uodo-blue-60) !important;
        border-bottom-color: var(--uodo-blue-60) !important;
    }

    div[data-testid="stExpander"] {
        border: 1px solid var(--uodo-blue-30) !important;
        border-radius: var(--uodo-border-radius) !important;
    }

    .filter-label {
        font-size: 0.78rem; font-weight: 700; color: var(--uodo-blue-80);
        text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 4px;
    }

    /* ── Metryki w sidebarze ── */
    [data-testid="stSidebar"] [data-testid="stMetric"] label,
    [data-testid="stSidebar"] [data-testid="stMetric"] [data-testid="stMetricLabel"] p {
        color: #c5d3e8 !important;
        font-size: 0.78rem !important;
    }
    [data-testid="stSidebar"] [data-testid="stMetric"] [data-testid="stMetricValue"],
    [data-testid="stSidebar"] [data-testid="stMetric"] [data-testid="stMetricValue"] p {
        color: #ffffff !important;
        font-size: 1.4rem !important;
        font-weight: 700 !important;
    }
    [data-testid="stSidebar"] [data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.08);
        border-radius: 4px;
        padding: 6px 10px;
        margin-bottom: 6px;
    }
</style>
"""

PAGE_HEADER_HTML = """
<div class="page-header">
  <div class="page-header-inner">
    <div>
      <h1>Portal Orzeczeń UODO</h1>
      <div class="page-header-sub">Wyszukiwarka decyzji Prezesa UODO · Ustawa o ochronie danych osobowych · RODO</div>
    </div>
  </div>
</div>
"""


# ─────────────────────────── BUDOWANIE KONTEKSTU LLM ─────────────

_FRAGMENT_STOPWORDS = {
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


def _extract_fragment(content: str, query: str, max_len: int = 2000) -> str:
    """Wybiera najbardziej trafny fragment treści decyzji względem zapytania."""
    if not content or len(content) <= max_len:
        return content
    keywords = [
        w.lower()
        for w in re.split(r"\W+", query)
        if w.lower() not in _FRAGMENT_STOPWORDS and len(w) > 2
    ]
    if not keywords:
        return content[:max_len]
    step = 150
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


def build_context(
    docs: list[dict[str, Any]],
    query: str,
    max_chars: int = 18000,
    filters: dict[str, Any] | None = None,
    memory: AgentMemory | None = None,
) -> str:
    """Buduje kontekst dla LLM — szablony Jinja2 + priorytetyzacja decyzji UODO."""
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

    filter_note = (
        "UWAGA: Wyniki zawężone filtrami: " + "; ".join(filter_lines) + ".\n"
        if filter_lines
        else ""
    )

    memory_note = ""
    if memory:
        related = memory.find_related(query)
        if related:
            snippets = [
                f"- Poprzednie pytanie: «{e.query}» → znalezione decyzje: "
                + (", ".join(e.top_signatures[:3]) if e.top_signatures else "brak")
                for e in related[:2]
            ]
            memory_note = (
                "KONTEKST Z POPRZEDNICH ANALIZ (tej sesji):\n"
                + "\n".join(snippets)
                + "\n"
            )

    header = TPL_HEADER.render(
        query=query, filter_note=filter_note, memory_note=memory_note
    )
    parts = [header]
    chars = len(header)

    # Decyzje UODO pierwsze, RODO ostatnie
    docs_sorted = sorted(
        docs,
        key=lambda d: (
            CONTEXT_TYPE_ORDER.get(d.get("doc_type", ""), 9),
            -d.get("_score", 0),
        ),
    )

    for i, doc in enumerate(docs_sorted, 1):
        dtype = doc.get("doc_type", "")

        if dtype == "legal_act_article":
            chunk_idx = doc.get("chunk_index", 0)
            total = doc.get("chunk_total", 1)
            suffix = f"(część {chunk_idx + 1}/{total})" if total > 1 else ""
            block = TPL_ACT_ARTICLE.render(
                rank=i,
                art_num=doc.get("article_num", "?"),
                label_suffix=suffix,
                text=doc.get("content_text", ""),
            )
        elif dtype in ("gdpr_article", "gdpr_recital"):
            art_num = doc.get("article_num", "?")
            prefix = "Motyw" if dtype == "gdpr_recital" else f"Art. {art_num}"
            block = TPL_GDPR.render(
                rank=i, prefix=prefix, text=doc.get("content_text", "")
            )
        else:
            keywords = doc.get("keywords_text", "") or ", ".join(
                doc.get("keywords", [])
            )
            acts = doc.get("related_acts", [])[:4] + doc.get("related_eu_acts", [])[:2]
            block = TPL_DECISION.render(
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


# ─────────────────────────── KARTY WYNIKÓW ───────────────────────


def decision_url(doc: dict[str, Any]) -> str:
    sig = doc.get("signature", "")
    url = doc.get("source_url", "")
    if url:
        return url
    import re as _re

    slug = sig.lower().replace(".", "_")
    year_m = _re.search(r"\b(20\d{2})\b", sig)
    year = year_m.group(1) if year_m else "2024"
    return f"{UODO_PORTAL_BASE}/urn:ndoc:gov:pl:uodo:{year}:{slug}/content"


def render_decision_card(doc: dict[str, Any], rank: int) -> None:
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
    taxonomy_values = {
        v.lower()
        for v in doc.get("term_decision_type", []) + doc.get("term_sector", [])
    }
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

    graph_badge = (
        f' <span class="status-badge status-unknown">↗ {graph_rel or "graf"}</span>'
        if source == "graph"
        else ""
    )

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

    with st.container():
        if kw_list:
            shown = kw_list[:8]
            rest = len(kw_list) - len(shown)
            tags = " · ".join(f"`{k}`" for k in shown)
            suffix = f" *+{rest} więcej*" if rest > 0 else ""
            st.caption(f"🏷️ {tags}{suffix}")
        all_acts = doc.get("related_acts", [])[:4] + doc.get("related_eu_acts", [])[:2]
        if all_acts:
            st.caption("📜 Powołane akty: " + " · ".join(f"`{a}`" for a in all_acts))
        if graph_rel:
            st.caption(f"↗ powiązana przez graf: *{graph_rel}*")
    st.divider()


def render_act_article_card(doc: dict[str, Any], rank: int) -> None:
    art_num = doc.get("article_num", "?")
    chunk_idx = doc.get("chunk_index", 0)
    total = doc.get("chunk_total", 1)
    score = doc.get("_score", 0)
    text = doc.get("content_text", "")[:600]
    label = f"Art. {art_num} u.o.d.o." + (
        f" (część {chunk_idx + 1}/{total})" if total > 1 else ""
    )

    st.markdown(
        f"""
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
      <footer><small class="text-muted">score: {score:.3f}</small></footer>
    </article>""",
        unsafe_allow_html=True,
    )


def render_gdpr_card(doc: dict[str, Any], rank: int) -> None:
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
    chapter_html = (
        f'<small class="text-muted">Rozdział {chapter} — {chapter_title}</small>'
        if chapter and chapter_title
        else ""
    )

    st.markdown(
        f"""
    <article class="doc-list-item">
      <header>
        <a href="{GDPR_URL}" target="_blank">{label}</a>
        <span class="status-badge status-final">{badge_txt}</span>
      </header>
      <main>
        <h2>{chapter_html}</h2>
        <p>{text}{"…" if len(doc.get("content_text", "")) > 500 else ""}</p>
      </main>
      <footer><small class="text-muted">score: {score:.3f}</small></footer>
    </article>""",
        unsafe_allow_html=True,
    )


def render_card(doc: dict[str, Any], rank: int) -> None:
    """Dispatcher — wybiera typ karty na podstawie doc_type."""
    dtype = doc.get("doc_type", "")
    if dtype == "legal_act_article":
        render_act_article_card(doc, rank)
    elif dtype in ("gdpr_article", "gdpr_recital"):
        render_gdpr_card(doc, rank)
    else:
        render_decision_card(doc, rank)
