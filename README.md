# 🔐 UODO RAG — Wyszukiwarka Decyzji i Przepisów

Aplikacja RAG (Retrieval-Augmented Generation) do przeszukiwania decyzji Prezesa Urzędu Ochrony Danych Osobowych oraz przepisów ustawy o ochronie danych osobowych.

## Funkcje

- **Wyszukiwanie semantyczne** — model embeddingowy rozumie sens pytania, nie tylko słowa kluczowe
- **Graf powiązań** — decyzje UODO połączone siecią cytowań; wyszukiwanie rozszerza wyniki o powiązane orzeczenia
- **Analiza AI** — LLM syntetyzuje odpowiedź z konkretnymi odniesieniami do sygnatur i artykułów ustawy
- **Wyszukiwanie po tagach** — LLM automatycznie dobiera pasujące słowa kluczowe z bazy, obsługuje formy fleksyjne
- **Fast path po sygnaturze** — wpisanie sygnatury (np. `DKN.5110.16.2022`) trafia bezpośrednio do decyzji
- **Dwa typy dokumentów** — decyzje UODO + pełny tekst ustawy o ochronie danych osobowych (Dz.U. 2019 poz. 1781)

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

Przed uruchomieniem aplikacji należy zaindeksować dane:

```bash
# 1. Pobierz decyzje UODO z API portalu orzeczeń
python uodo_scraper.py --output uodo_decisions.jsonl

# 2. Zaindeksuj decyzje w Qdrant
python uodo_indexer.py --jsonl uodo_decisions.jsonl

# 3. Zaindeksuj ustawę o ochronie danych osobowych
python uodo_act_indexer.py --md D20191781L.md
```

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
├── requirements.txt     # Zależności Python
├── .env                 # Klucze API (nie commitować!)
└── uodo_graph.pkl       # Graf powiązań (generowany automatycznie)
```

## Źródła danych

- **Decyzje UODO** — [orzeczenia.uodo.gov.pl](https://orzeczenia.uodo.gov.pl) (REST API)
- **Ustawa o ochronie danych osobowych** — [Dz.U. 2019 poz. 1781](https://isap.sejm.gov.pl/isap.nsf/DocDetails.xsp?id=WDU20190001781)

## Model embeddingowy

Aplikacja wykorzystuje **[sdadas/mmlw-retrieval-roberta-large](https://huggingface.co/sdadas/mmlw-retrieval-roberta-large)** — polski model semantyczny zoptymalizowany do wyszukiwania.

## Modele LLM

| Provider | Zalecany model | Uwagi |
|---|---|---|
| Ollama Cloud | `gpt-oss:120b` | Domyślny, najlepsza jakość |
| Groq | `llama-3.3-70b-versatile` | Szybki, darmowy limit |
