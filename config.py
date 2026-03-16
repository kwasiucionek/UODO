"""
Konfiguracja aplikacji UODO RAG — stałe i zmienne środowiskowe.
"""

import os
import re

# Wczytaj .env jeśli istnieje
try:
    from dotenv import load_dotenv
    _ = load_dotenv()
except ImportError:
    pass

# ── Qdrant ────────────────────────────────────────────────────────
QDRANT_URL       = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME  = "uodo_decisions"
GRAPH_PATH       = os.getenv("UODO_GRAPH_PATH", "./uodo_graph.pkl")
EMBED_MODEL      = os.getenv("EMBED_MODEL", "sdadas/mmlw-retrieval-roberta-large")

# ── LLM providerzy ───────────────────────────────────────────────
GROQ_API_KEY        = os.getenv("GROQ_API_KEY", "")
OLLAMA_CLOUD_API_KEY = os.getenv("OLLAMA_CLOUD_API_KEY", "")
OLLAMA_CLOUD_URL    = os.getenv("OLLAMA_CLOUD_URL", "https://ollama.com")
OLLAMA_LOCAL_URL    = os.getenv("OLLAMA_LOCAL_URL", "http://localhost:11434")

PROVIDERS           = ["Ollama Cloud", "Groq"]
DEFAULT_PROVIDER    = "Ollama Cloud"
DEFAULT_OLLAMA_MODEL = "gpt-oss:120b"
DEFAULT_GROQ_MODEL  = "openai/gpt-oss-120b"

# ── Wyszukiwanie ─────────────────────────────────────────────────
TOP_K        = 8
GRAPH_DEPTH  = 2
MAX_ACT_DOCS  = 5   # maks. artykułów u.o.d.o. w wynikach
MAX_GDPR_DOCS = 3   # maks. artykułów RODO w wynikach

# ── URL-e zewnętrzne ─────────────────────────────────────────────
UODO_PORTAL_BASE = "https://orzeczenia.uodo.gov.pl/document"
ISAP_ACT_URL     = "https://isap.sejm.gov.pl/isap.nsf/DocDetails.xsp?id=WDU20190001781"
GDPR_URL         = "https://eur-lex.europa.eu/legal-content/PL/TXT/?uri=CELEX:32016R0679"

# ── Regex: sygnatura UODO wpisana bezpośrednio jako query ────────
RE_QUERY_SIG = re.compile(r"^\s*([A-Z]{2,6}\.\d{3,5}\.\d+\.\d{4})\s*$", re.IGNORECASE)

# ── Stopwords do ekstrakcji fraz z zapytania ─────────────────────
QUERY_STOPWORDS = {
    "jakie", "są", "w", "o", "i", "z", "do", "na", "co", "ile", "jak",
    "czy", "przez", "dla", "po", "przy", "od", "ze", "to", "a", "że",
    "się", "nie", "być", "który", "które", "która",
}

# ── Taksonomia portalu UODO ───────────────────────────────────────
TAXONOMY_STATIC: dict[str, list[str]] = {
    "term_decision_type": ["nakaz", "odmowa", "umorzenie", "upomnienie", "inne"],
    "term_sector": [
        "BIP", "DODO", "Finanse", "Marketing", "Mieszkalnictwo", "Monitoring",
        "Pozostałe", "Szkolnictwo", "Telekomunikacja", "Ubezpieczenia",
        "Zatrudnienie", "Zdrowie",
    ],
    "term_corrective_measure": [
        "ostrzeżenie", "upomnienie", "nakaz spełnienia żądania", "dostosowanie",
        "poinformowanie", "ograniczenie przetwarzania",
        "sprostowanie/usunięcie/ograniczenie", "cofnięcie certyfikacji",
        "administracyjna kara pieniężna", "państwo trzecie",
    ],
    "term_violation_type": [],   # wypełniane dynamicznie z Qdrant
    "term_legal_basis":    [],   # j.w.
}
