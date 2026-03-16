# 🔐 UODO RAG — Wyszukiwarka Decyzji i Przepisów

Aplikacja RAG (Retrieval-Augmented Generation) do przeszukiwania decyzji Prezesa Urzędu Ochrony Danych Osobowych oraz przepisów ustawy o ochronie danych osobowych i rozporządzenia RODO.

Wygląd i filtry wzorowane na oficjalnym [Portalu Orzeczeń UODO](https://orzeczenia.uodo.gov.pl).

## Funkcje

- **Wyszukiwanie po tagach (bez LLM)** — frazy bezpośrednio z zapytania są dopasowywane do tagów bazy; szuka "dane genetyczne" jako całej frazy, nie osobno "dane" i "genetyczne"
- **Wyszukiwanie semantyczne** — model embeddingowy rozumie sens pytania, nie tylko słowa kluczowe; uruchamiany jako fallback gdy fraza nie jest tagiem
- **Graf powiązań** — decyzje UODO połączone siecią cytowań; wyszukiwanie rozszerza wyniki o powiązane orzeczenia
- **Analiza AI** — LLM syntetyzuje odpowiedź z konkretnymi odniesieniami do sygnatur i artykułów ustawy
- **Priorytety wyników** — decyzje UODO zawsze na pierwszym miejscu (bez limitu liczby), u.o.d.o. max 5, RODO max 3
- **Fast path po sygnaturze** — wpisanie sygnatury (np. `DKN.5110.16.2022`) trafia bezpośrednio do decyzji
- **Trzy typy dokumentów** — decyzje UODO + ustawa o ochronie danych osobowych + RODO (artykuły i motywy)
- **Filtry taksonomiczne** — rodzaj decyzji, rodzaj naruszenia, podstawa prawna, środek naprawczy, sektor
- **Pamięć epizodyczna** — historia sesji z poprzednimi pytaniami i znalezionymi decyzjami
- **Reasoning Step** — LLM dekompozycja dłuższych pytań przed wyszukiwaniem

## Architektura wyszukiwania

```
zapytanie użytkownika
        │
        ├─► 1. Frazy z zapytania → dopasowanie do tagów w bazie (BEZ LLM)
        │         └─► keyword_exact_search → WSZYSTKIE decyzje z tagiem
        │
        ├─► 2. Tagi LLM (fallback — tylko gdy 1. nic nie znalazło)
        │         └─► keyword_exact_search → WSZYSTKIE decyzje z tagiem
        │
        ├─► 3. Semantic search (fallback — tylko gdy tagi nic nie dały)
        │         └─► top_k=20, score_threshold=0.45
        │
        ├─► Graf cytowań (NetworkX) → rozszerzenie o powiązane decyzje
        │
        └─► LLM (Ollama Cloud / Groq) → odpowiedź z odniesieniami
```

## Baza dokumentów

| Typ | Źródło | Liczba |
|---|---|---|
| Decyzje UODO | [orzeczenia.uodo.gov.pl](https://orzeczenia.uodo.gov.pl) | ~560 |
| Ustawa o ochronie danych (u.o.d.o.) | [Dz.U. 2019 poz. 1781](https://isap.sejm.gov.pl/isap.nsf/DocDetails.xsp?id=WDU20190001781) | artykuły 1–108 |
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
python tools/uodo_scraper.py --output tools/uodo_decisions.jsonl

# Wzbogać o pola taksonomiczne (rodzaj decyzji, naruszenie, sektor itd.)
python tools/enrich_jsonl_taxonomy.py \
    --input tools/uodo_decisions.jsonl \
    --output tools/uodo_decisions_enriched.jsonl

# Zaindeksuj w Qdrant
python tools/uodo_indexer.py --jsonl tools/uodo_decisions_enriched.jsonl --rebuild
```

### 2. Ustawa o ochronie danych osobowych

```bash
python tools/uodo_act_indexer.py --md tools/D20191781L.md
```

### 3. RODO (rozporządzenie UE 2016/679)

```bash
# Z lokalnego pliku Markdown
python tools/rodo_indexer.py --md tools/rodo_2016_679_pl.md

# Dry run — parsuj i wypisz bez indeksowania
python tools/rodo_indexer.py --md tools/rodo_2016_679_pl.md --dry-run
```

### 4. Słowa kluczowe dla artykułów (opcjonalne)

Generuje tagi dla artykułów u.o.d.o. i RODO przez LLM, żeby były widoczne w filtrach:

```bash
python tools/enrich_act_keywords.py --provider ollama --model qwen3:14b
python tools/enrich_act_keywords.py --provider groq --model llama-3.3-70b-versatile
python tools/enrich_act_keywords.py --dry-run   # tylko podgląd, nie zapisuje
```

## Uruchomienie

```bash
streamlit run main.py
```

Aplikacja dostępna pod adresem: http://localhost:8501

## Struktura projektu

```
.
├── main.py                      # Główna aplikacja Streamlit (punkt wejścia)
├── config.py                    # Stałe i zmienne środowiskowe
├── models.py                    # Modele Pydantic + szablony Jinja2
├── search.py                    # Qdrant, graf, hybrid search, tagi, taksonomia
├── llm.py                       # Wywołania LLM, dekompozycja zapytań, lista modeli
├── ui.py                        # Karty wyników, build_context, CSS
│
├── tools/
│   ├── uodo_scraper.py          # Scraper decyzji z API portalu UODO
│   ├── uodo_indexer.py          # Indeksowanie decyzji w Qdrant
│   ├── uodo_act_indexer.py      # Indeksowanie ustawy o ochronie danych
│   ├── rodo_indexer.py          # Indeksowanie RODO (2016/679) z pliku Markdown
│   ├── enrich_jsonl_taxonomy.py # Wzbogacenie JSONL o pola taksonomiczne
│   ├── enrich_act_keywords.py   # Generowanie słów kluczowych dla artykułów (LLM)
│   ├── eval.py                  # Automatyczna ewaluacja systemu (binarny leaderboard)
│   ├── D20191781L.md            # Tekst ustawy o ochronie danych osobowych
│   ├── rodo_2016_679_pl.md      # Tekst RODO w języku polskim
│   ├── uodo_decisions.jsonl     # Surowe decyzje z API portalu UODO
│   └── uodo_decisions_enriched.jsonl  # Decyzje z polami taksonomicznymi
│
├── requirements.txt
├── .env                         # Klucze API (nie commitować!)
├── .gitignore
└── uodo_graph.pkl               # Graf powiązań (generowany automatycznie)
```

## Model embeddingowy

Aplikacja wykorzystuje **[sdadas/mmlw-retrieval-roberta-large](https://huggingface.co/sdadas/mmlw-retrieval-roberta-large)** — polski model semantyczny zoptymalizowany do wyszukiwania.

## Modele LLM

| Provider | Domyślny model | Uwagi |
|---|---|---|
| Ollama Cloud | `gpt-oss:120b` | Domyślny, najlepsza jakość |
| Groq | `openai/gpt-oss-120b` | Szybki, darmowy limit |

## Filtry taksonomiczne

Filtry dostępne wyłącznie dla decyzji UODO (ignorowane przy wyszukiwaniu w u.o.d.o. i RODO):

| Filtr | Opis |
|---|---|
| Rodzaj decyzji | nakaz, odmowa, umorzenie, upomnienie, inne |
| Rodzaj naruszenia | brak podstawy prawnej, niezgłoszenie naruszenia, brak IOD, … |
| Podstawa prawna | zgoda, umowa, obowiązek prawny, uzasadniony interes, … |
| Środek naprawczy | ostrzeżenie, nakaz spełnienia żądania, kara pieniężna, … |
| Sektor | marketing, zdrowie, szkolnictwo, finanse, telekomunikacja, … |

## Ewaluacja

Automatyczna ewaluacja systemu z 10 złotymi pytaniami i binarnym leaderboardem:

```bash
python tools/eval.py                  # pełna ewaluacja
python tools/eval.py --question 3     # tylko pytanie nr 3
python tools/eval.py --verbose        # z pełnymi odpowiedziami
```

Wyniki zapisywane są do `eval_results.json`.
