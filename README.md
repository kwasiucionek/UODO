# 🔐 UODO RAG — Wyszukiwarka Decyzji i Przepisów

Aplikacja RAG (Retrieval-Augmented Generation) do przeszukiwania decyzji Prezesa Urzędu Ochrony Danych Osobowych oraz przepisów ustawy o ochronie danych osobowych i rozporządzenia RODO.

## Funkcje

- **Wyszukiwanie semantyczne** — model embeddingowy rozumie sens pytania, nie tylko słowa kluczowe
- **Graf powiązań** — decyzje UODO połączone siecią cytowań; wyszukiwanie rozszerza wyniki o powiązane orzeczenia
- **Analiza AI** — LLM syntetyzuje odpowiedź z konkretnymi odniesieniami do sygnatur i artykułów ustawy
- **Wyszukiwanie po tagach** — LLM automatycznie dobiera pasujące słowa kluczowe z bazy, obsługuje formy fleksyjne
- **Fast path po sygnaturze** — wpisanie sygnatury (np. `DKN.5110.16.2022`) trafia bezpośrednio do decyzji
- **Trzy typy dokumentów** — decyzje UODO + ustawa o ochronie danych osobowych + RODO (artykuły i motywy)

## Architektura

```
zapytanie użytkownika
        │
        ├─► LLM dobiera tagi → filtr do Qdrant
        │
        ├─► Semantic search (mmlw-retrieval-roberta-large)
        │
        ├─► Graf cytowań (NetworkX) → rozszerzenie o powiązane decyzje
        │
        └─► LLM (Ollama Cloud / Groq) → odpowiedź z odniesieniami
```

## Baza dokumentów

| Typ | Źródło | Liczba |
|---|---|---|
| Decyzje UODO | [orzeczenia.uodo.gov.pl](https://orzeczenia.uodo.gov.pl) | ~560 |
| Ustawa o ochronie danych (u.o.d.o.) | [Dz.U. 2019 poz. 1781](https://isap.sejm.gov.pl/isap.nsf/DocDetails.xsp?id=WDU20190001781) | artykuły 1–110 |
| RODO — artykuły | [EUR-Lex 32016R0679](https://eur-lex.europa.eu/legal-content/PL/TXT/?uri=CELEX:32016R0679) | 99 artykułów |
| RODO — motywy | j.w. | 173 motywy |

## Wymagania systemowe

- Python 3.10+
- [Qdrant](https://qdrant.tech/) (lokalnie lub zdalnie)
- Klucz API: [Ollama Cloud](https://ollama.com) lub [Groq](https://console.groq.com)

## Instalacja

```bash
pip install -r requirements.txt
```

## Konfiguracja

Utwórz plik `.env` w katalogu aplikacji:

```env
# Ollama Cloud (zalecany)
OLLAMA_CLOUD_API_KEY=twoj_klucz

# lub Groq
GROQ_API_KEY=twoj_klucz

# Qdrant (domyślnie localhost:6333)
QDRANT_URL=http://localhost:6333

# Opcjonalne
UODO_GRAPH_PATH=./uodo_graph.pkl
EMBED_MODEL=sdadas/mmlw-retrieval-roberta-large
```

## Przygotowanie bazy danych

### 1. Decyzje UODO

```bash
# Pobierz decyzje z API portalu orzeczeń
python uodo_scraper.py --output uodo_decisions.jsonl

# Zaindeksuj w Qdrant
python uodo_indexer.py --jsonl uodo_decisions.jsonl
```

### 2. Ustawa o ochronie danych osobowych

```bash
python uodo_act_indexer.py --md D20191781L.md
```

### 3. RODO (rozporządzenie UE 2016/679)

```bash
# Pobierz i zaindeksuj automatycznie (PDF z EUR-Lex)
python rodo_indexer.py

# lub z lokalnego pliku PDF
python rodo_indexer.py --pdf rodo.pdf

# Test parsowania bez indeksowania
python rodo_indexer.py --dry-run
```

Skrypt `rodo_indexer.py` indeksuje osobno 99 artykułów (typ `gdpr_article`) oraz 173 motywy preambuły (typ `gdpr_recital`).

## Uruchomienie

```bash
streamlit run uodo_app.py
```

Aplikacja dostępna pod adresem: http://localhost:8501

## Struktura projektu

```
.
├── uodo_app.py          # Główna aplikacja Streamlit
├── uodo_scraper.py      # Scraper decyzji z API portalu UODO
├── uodo_indexer.py      # Indeksowanie decyzji w Qdrant
├── uodo_act_indexer.py  # Indeksowanie ustawy o ochronie danych
├── rodo_indexer.py      # Indeksowanie RODO (2016/679) z EUR-Lex
├── requirements.txt     # Zależności Python
├── .env                 # Klucze API (nie commitować!)
└── uodo_graph.pkl       # Graf powiązań (generowany automatycznie)
```

## Model embeddingowy

Aplikacja wykorzystuje **[sdadas/mmlw-retrieval-roberta-large](https://huggingface.co/sdadas/mmlw-retrieval-roberta-large)** — polski model semantyczny zoptymalizowany do wyszukiwania.

## Modele LLM

| Provider | Domyślny model | Uwagi |
|---|---|---|
| Ollama Cloud | `gpt-oss:120b` | Domyślny, najlepsza jakość |
| Groq | `openai/gpt-oss-120b` | Szybki, darmowy limit |
