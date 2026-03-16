# UODO RAG — Szczegółowy opis działania aplikacji

## Spis treści

1. [Cel i kontekst](#1-cel-i-kontekst)
2. [Architektura ogólna](#2-architektura-ogólna)
3. [Baza wiedzy i indeksowanie](#3-baza-wiedzy-i-indeksowanie)
4. [Przepływ danych — od zapytania do odpowiedzi](#4-przepływ-danych--od-zapytania-do-odpowiedzi)
5. [Moduł wyszukiwania (search.py)](#5-moduł-wyszukiwania-searchpy)
6. [Graf powiązań](#6-graf-powiązań)
7. [Moduł LLM (llm.py)](#7-moduł-llm-llmpy)
8. [Budowanie kontekstu (ui.py → build_context)](#8-budowanie-kontekstu-uipy--build_context)
9. [Interfejs użytkownika (main.py + ui.py)](#9-interfejs-użytkownika-mainpy--uipy)
10. [Pamięć epizodyczna](#10-pamięć-epizodyczna)
11. [Konfiguracja i modele danych](#11-konfiguracja-i-modele-danych)
12. [Narzędzia pomocnicze (tools/)](#12-narzędzia-pomocnicze-tools)
13. [Kluczowe decyzje projektowe](#13-kluczowe-decyzje-projektowe)

---

## 1. Cel i kontekst

Aplikacja jest systemem **RAG (Retrieval-Augmented Generation)** — to podejście do budowania asystentów AI, w którym model językowy (LLM) odpowiada na pytania **wyłącznie na podstawie dostarczonych mu dokumentów**, a nie na podstawie własnej ogólnej wiedzy. Dzięki temu odpowiedzi są ugruntowane w konkretnych aktach prawnych i orzeczeniach, a nie w tym, czego model nauczył się podczas treningu.

**Problem, który rozwiązuje:** Baza decyzji Prezesa UODO liczy ponad 560 orzeczeń. Tradycyjne wyszukiwanie pełnotekstowe jest niedokładne (nie rozumie sensu pytania), a przeglądanie ręczne jest bardzo czasochłonne. Aplikacja łączy wyszukiwanie semantyczne, filtrowanie po tagach i syntezę odpowiedzi przez LLM.

**Trzy źródła wiedzy:**
- **Decyzje UODO** — orzeczenia administracyjne Prezesa Urzędu Ochrony Danych Osobowych (~560 dokumentów)
- **Ustawa o ochronie danych osobowych (u.o.d.o.)** — artykuły 1–108, Dz.U. 2019 poz. 1781
- **RODO** — 99 artykułów i 173 motywy preambuły rozporządzenia (UE) 2016/679

---

## 2. Architektura ogólna

Aplikacja składa się z sześciu modułów Python oraz zestawu narzędzi:

```
┌─────────────────────────────────────────────────────────────┐
│                        main.py                              │
│              (punkt wejścia, logika UI, routing)            │
└──────┬───────────┬──────────────┬──────────────┬────────────┘
       │           │              │              │
       ▼           ▼              ▼              ▼
  config.py    models.py      search.py       llm.py
  (stałe,      (Pydantic,     (Qdrant,        (LLM API,
  env vars)    Jinja2)        graf, tagi)     dekompozycja)
                                    │
                              ui.py (karty,
                              kontekst LLM,
                              CSS)
```

**Zewnętrzne zależności:**
- **Qdrant** — wektorowa baza danych przechowująca embeddingi dokumentów i ich metadane (payloady)
- **SentenceTransformers** — lokalny model embeddingowy `sdadas/mmlw-retrieval-roberta-large` (polski)
- **NetworkX** — biblioteka do grafu powiązań między decyzjami
- **Ollama Cloud / Groq** — zewnętrzne API LLM do generowania odpowiedzi i dekompozycji zapytań
- **Streamlit** — framework webowy dla interfejsu użytkownika

---

## 3. Baza wiedzy i indeksowanie

Zanim aplikacja zacznie działać, dokumenty muszą zostać przetworzone i zaindeksowane w Qdrant. To jednorazowy (lub rzadki) proces wykonywany przez narzędzia z katalogu `tools/`.

### 3.1 Struktura kolekcji Qdrant

Wszystkie dokumenty trafiają do **jednej kolekcji** o nazwie `uodo_decisions`. Każdy punkt (dokument) w Qdrant ma:

- **Wektor** — embedding o wymiarze ~1024 (zależnie od modelu), reprezentujący semantyczne znaczenie treści
- **Payload** — słownik metadanych dostępny do filtrowania i wyświetlania

Kluczowe pola payloadu:

| Pole | Typ | Opis |
|------|-----|------|
| `doc_type` | keyword | `uodo_decision` / `legal_act_article` / `gdpr_article` / `gdpr_recital` |
| `signature` | keyword | Sygnatura decyzji, np. `DKN.5110.16.2022` |
| `keywords` | keyword[] | Lista tagów, np. `["dane genetyczne", "dane szczególnych kategorii"]` |
| `status` | keyword | `prawomocna` / `nieprawomocna` / `uchylona` |
| `year` | integer | Rok decyzji |
| `content_text` | text | Pełna treść dokumentu (do 50 000 znaków) |
| `term_decision_type` | keyword[] | Taksonomia: rodzaj decyzji |
| `term_violation_type` | keyword[] | Taksonomia: rodzaj naruszenia |
| `term_legal_basis` | keyword[] | Taksonomia: podstawa prawna |
| `term_corrective_measure` | keyword[] | Taksonomia: środek naprawczy |
| `term_sector` | keyword[] | Taksonomia: sektor |
| `related_uodo_rulings` | text[] | Sygnatury cytowanych decyzji UODO |
| `related_acts` | text[] | Cytowane polskie akty prawne |
| `related_eu_acts` | text[] | Cytowane akty UE |

### 3.2 Indeksowanie decyzji UODO (uodo_scraper.py + uodo_indexer.py)

**Etap 1 — Scraping:** `uodo_scraper.py` pobiera decyzje przez REST API portalu `orzeczenia.uodo.gov.pl`. Dla każdej decyzji pobiera:
- Treść pełną (endpoint `/body.txt`)
- Metadane: tytuł, tagi, podmioty, rodzaj decyzji (`/meta.json`)
- Daty ogłoszenia i publikacji (`/dates.json`)
- Powiązania z innymi aktami (`refs` z meta.json + ekstrakcja z treści regexem)

Powiązania są wyciągane dwutorowo: z pola `refs` w meta.json (jeśli API je zwraca) oraz fallbackiem przez wyrażenia regularne z treści decyzji — wzorce dla `Dz.U. XXXX poz. XXXX`, `EU 2016/679`, sygnatur UODO i wyroków NSA/WSA.

**Etap 2 — Wzbogacenie taksonomii:** `enrich_jsonl_taxonomy.py` mapuje pole `meta.terms[]` z API (numerowane etykiety jak `1.1`, `2.3`, `9.1`) na konkretne pola semantyczne (`term_decision_type`, `term_violation_type` itd.). Prefix numeru etykiety determinuje kategorię.

**Etap 3 — Indeksowanie:** `uodo_indexer.py` buduje tekst do embeddingu (sygnatura + tytuł + tagi + fragment treści + powołane akty), generuje wektor przez model SentenceTransformers i zapisuje punkt do Qdrant. UUID punktu jest deterministyczny — wyznaczany przez MD5 sygnatury.

### 3.3 Indeksowanie ustawy (uodo_act_indexer.py)

Ustawa dostarczona jest jako plik Markdown (`D20191781L.md`). Parser rozpoznaje nagłówki artykułów (`Art. X.`) i wyodrębnia ich treść. Artykuły dłuższe niż 3000 znaków są dzielone na chunki z overlapem 300 znaków, żeby kontekst nie był urwany w połowie zdania.

Każdy chunk otrzymuje `doc_type = "legal_act_article"` i unikalne `doc_id` w formacie `uodo_act:Dz.U. 2019 poz. 1781:artX:chunkY`.

### 3.4 Indeksowanie RODO (rodo_indexer.py)

Plik `rodo_2016_679_pl.md` zawiera pełny tekst RODO. Parser rozróżnia dwa typy:
- **Motywy** — linie zaczynające się od `- (N) treść`, typ `gdpr_recital`
- **Artykuły** — nagłówki `# Artykuł N`, typ `gdpr_article`

Długie artykuły również są chunkowane (max 1200 znaków, overlap 100 znaków).

---

## 4. Przepływ danych — od zapytania do odpowiedzi

Poniżej pełny przepływ dla typowego zapytania, np. "Jakie decyzje dotyczą danych genetycznych?":

```
Użytkownik wpisuje zapytanie
            │
            ▼
    [1] Reasoning Step (llm.py)
    LLM dekompozycja pytania → QueryDecomposition
    (tylko dla zapytań > 3 słów i gdy AI włączone)
            │
            ▼
    [2] Hybrid Search (search.py)
    ┌─────────────────────────────────────┐
    │ 1a. Explicit keyword z UI?          │
    │     → keyword_exact_search          │
    │ 1b. Frazy z query → tagi w bazie?   │
    │     "dane genetyczne" → tag match   │
    │     → keyword_exact_search (scroll) │
    │ 1c. Tagi LLM (fallback)?            │
    │     → keyword_exact_search          │
    │ 1d. Semantic search (last resort)   │
    │ + u.o.d.o. semantic (max 5)         │
    │ + RODO semantic (max 3)             │
    │ + Graf cytowań                      │
    └─────────────────────────────────────┘
            │
            ▼
    [3] Build Context (ui.py)
    Sortowanie: decyzje → u.o.d.o. → RODO
    Jinja2 szablony dla każdego typu
    Limit 18 000 znaków
            │
            ▼
    [4] LLM Stream (llm.py)
    System prompt + kontekst → odpowiedź
    Streamowanie tokenów do UI
            │
            ▼
    [5] Wyświetlenie w Streamlit (main.py + ui.py)
    Odpowiedź AI + karty dokumentów w zakładkach
            │
            ▼
    [6] Zapis do pamięci epizodycznej (models.py)
    AgentMemory ← MemoryEntry(query, sygnatury, snippet)
```

---

## 5. Moduł wyszukiwania (search.py)

To serce aplikacji. Funkcja `hybrid_search()` realizuje wieloetapową strategię wyszukiwania z wyraźnymi priorytetami.

### 5.1 Semantic search

`semantic_search()` przyjmuje zapytanie, generuje jego embedding i pyta Qdrant o najbliższe wektory:

```python
def semantic_search(query, top_k=8, filters=None, score_threshold=0.25):
    vec = embed(query)          # SentenceTransformers → lista floatów
    res = client.query_points(  # ANN (Approximate Nearest Neighbor)
        query=vec,
        limit=top_k,
        query_filter=_build_qdrant_filter(filters),
        score_threshold=score_threshold,  # odcięcie nieistotnych wyników
    )
```

`score_threshold` jest konfigurowalny per wywołanie — dla decyzji UODO w fallbacku używamy `0.45` (rygorystyczny), dla u.o.d.o. `0.25`, dla RODO `0.3`. Wyższy próg = mniej, ale trafniejszych wyników.

`_build_qdrant_filter()` buduje obiekt `Filter(must=[...])` z Qdrant na podstawie słownika filtrów UI — statusu, roku, typów dokumentów, pól taksonomicznych itd.

### 5.2 Keyword exact search

`keyword_exact_search()` to scroll bez limitu liczby wyników — pobiera **wszystkie** dokumenty z danym tagiem:

```python
def keyword_exact_search(keyword, filters=None):
    # Dodaje keyword do filtrów jako FieldCondition na polu "keywords"
    kw_filters = {**filters, "keyword": keyword}
    # Scroll z paginacją po 100 punktów
    while True:
        pts, next_offset = client.scroll(scroll_filter=..., limit=100)
        docs.extend(pts)
        if next_offset is None: break
```

To kluczowa różnica względem `semantic_search` — tam jest `limit=top_k`, tutaj nie ma limitu. Dzięki temu dla zapytania "dane genetyczne" pobieramy wszystkie 26 decyzji, nie tylko top-8.

### 5.3 Logika hybrid_search — szczegółowo

Funkcja prowadzi trzy osobne buckety: `decisions`, `act_docs`, `gdpr_docs`. Każdy bucket jest wypełniany niezależnie.

**Bucket decyzji UODO (bez limitu):**

```
Krok 1a — explicit keyword z UI:
  Użytkownik wybrał tag z listy → keyword_exact_search
  Przykład: filtr "zgoda" → wszystkie decyzje z tagiem "zgoda"

Krok 1b — frazy bezpośrednio z zapytania (BEZ LLM):
  query = "dane genetyczne pacjentów szpitala"
  words = ["dane", "genetyczne", "pacjentów", "szpitala"]
  frazy = ["dane", "genetyczne", "pacjentów", "szpitala",
           "dane genetyczne", "genetyczne pacjentów", "pacjentów szpitala"]
  
  all_tags_lower = {t.lower(): t for t in get_all_tags()}
  # Sprawdza po kolei:
  # "dane" → nie jest tagiem
  # "genetyczne" → nie jest tagiem
  # "dane genetyczne" → JEST tagiem! → keyword_exact_search("dane genetyczne")
  # Wynik: 26 decyzji z tagiem "dane genetyczne"

Krok 1c — tagi LLM (fallback gdy 1b nic nie znalazło):
  Wywołuje LLM z listą dostępnych tagów → LLM proponuje pasujące
  Używane gdy zapytanie jest opisowe i nie zawiera żadnej frazy będącej tagiem
  Przykład: "przetwarzanie danych w celach marketingowych bez zgody"
  LLM może zaproponować: "marketing", "zgoda", "brak podstawy prawnej"

Krok 1d — semantic search (ostatni fallback):
  Uruchamiany TYLKO gdy decisions < 5
  top_k=20, score_threshold=0.45
  Łapie przypadki gdzie fraza w ogóle nie jest tagiem
```

**Bucket u.o.d.o. (max 5 artykułów):**
- Najpierw explicit keyword (jeśli ustawiony w UI)
- Uzupełnienie semantic searchem do limitu 5

**Bucket RODO (max 3 artykuły):**
- Analogicznie jak u.o.d.o., ale próg semantyczny wyższy (0.3)

**Dlaczego taka kolejność?** Tagi są precyzyjne — "dane genetyczne" to konkretna etykieta przypisana przez ekspertów podczas indeksowania decyzji. Semantic search rozumie sens, ale może zwrócić decyzje o "danych biometrycznych" albo "danych szczególnych kategorii" gdy pytamy o "dane genetyczne" — bo wektory tych fraz są blisko siebie. Frazy bezpośrednie z zapytania są trafniejsze i nie wymagają wywołania LLM.

### 5.4 Deduplikacja

Funkcja `doc_key()` generuje unikalny klucz dla każdego dokumentu. Zbiór `seen_keys` pilnuje, żeby ten sam dokument nie trafił do wyników dwukrotnie — nawet jeśli pasuje do kilku tagów jednocześnie.

---

## 6. Graf powiązań

Graf jest budowany raz (przy pierwszym uruchomieniu) i zapisywany do pliku `uodo_graph.pkl`. Przy kolejnych uruchomieniach jest wczytywany z dysku.

### 6.1 Budowanie grafu

```python
G = nx.DiGraph()  # graf skierowany
# Węzły = sygnatury decyzji
# Krawędzie = cytowania między decyzjami
# Typy relacji: CITES_UODO, CITES_ACT, CITES_EU
```

Dla każdej decyzji w Qdrant tworzony jest węzeł, a pole `related_uodo_rulings` (lista sygnatur cytowanych decyzji) zamieniane jest na krawędzie skierowane.

### 6.2 Rozszerzanie wyników

Po znalezieniu decyzji przez wyszukiwanie (`seed_sigs`), `graph_expand()` przechodzi po grafie w dwóch kierunkach:
- **Następniki** (decyzje cytowane przez znalezione) — relacja "cytowana", waga 0.6
- **Poprzedniki** (decyzje, które cytują znalezione) — relacja "cytuje tę decyzję", waga 0.5

Waga maleje z głębią grafu (`decay = 0.65^d`), żeby decyzje bezpośrednio powiązane miały wyższy priorytet niż te powiązane przez pośrednika. Maksymalnie 15 dodatkowych decyzji z grafu, maksymalna głębokość 2.

Rozszerzenie grafu ma sens, bo jeśli znaleziona decyzja jest precedensem, to decyzje ją cytujące są prawdopodobnie tematycznie powiązane — nawet jeśli nie zawierają szukanej frazy.

---

## 7. Moduł LLM (llm.py)

### 7.1 Reasoning Step — dekompozycja zapytania

Przed wyszukiwaniem, dla zapytań dłuższych niż 3 słowa, LLM analizuje pytanie i generuje strukturę `QueryDecomposition`:

```python
class QueryDecomposition(BaseModel):
    query_type: QueryType          # szukam_decyzji / szukam_przepisu / analiza_ogólna / pytanie_faktyczne
    search_keywords: list[str]     # synonimy prawne (max 5)
    gdpr_articles_hint: list[str]  # np. ["Art. 9", "Art. 17"]
    uodo_act_articles_hint: list[str]
    year_from_hint: int | None     # zawężenie dat
    year_to_hint: int | None
    enriched_query: str            # rozszerzone zapytanie do semantic search
    reasoning: str                 # uzasadnienie (widoczne w UI)
```

Prompt do LLM wymaga odpowiedzi wyłącznie w formacie JSON. Wynik jest parsowany przez Pydantic — jeśli parsowanie się nie powiedzie, używane jest oryginalne zapytanie.

`enriched_query` trafia do semantic search zamiast oryginalnego zapytania — np. "dane genetyczne" może zostać rozszerzone do "przetwarzanie danych genetycznych pacjentów ochrona zdrowia RODO Art. 9".

`year_from_hint` i `year_to_hint` są automatycznie dodawane do filtrów Qdrant, jeśli użytkownik nie ustawił dat ręcznie.

### 7.2 Dobór tagów przez LLM

`extract_tags_with_llm()` wysyła do LLM listę wszystkich tagów z bazy i prosi o wybór pasujących. Odpowiedź jest parsowana linijka po linijce — LLM może wskazać istniejące tagi (dokładna pisownia) lub zaproponować nowe (z prefiksem `[NOWY]`).

Nowe tagi (spoza listy) mogą trafić do wyszukiwania, ale w Qdrant nie znajdą żadnych dokumentów — są użyteczne tylko jako sygnał dla semantic search. Funkcja zwraca max 8 tagów z listy + 2 nowe.

Ta funkcja jest wywoływana zawsze, ale wynik jest używany **tylko jako fallback** (krok 1c) — gdy bezpośrednie dopasowanie fraz z zapytania nic nie znalazło.

### 7.3 Streamowanie odpowiedzi

`call_llm_stream()` generuje tokeny asynchronicznie. W przypadku Ollama Cloud parsuje linie NDJSON (każda linia to JSON z polem `message.content`). W przypadku Groq używa oficjalnego SDK z `stream=True`.

System prompt instruuje LLM, żeby:
- odpowiadał wyłącznie po polsku
- zawsze cytował sygnatury decyzji i numery artykułów
- wyraźnie poinformował, jeśli kontekst nie zawiera odpowiedzi

### 7.4 JSON bez streamowania

`call_llm_json()` służy do wywołań wymagających strukturyzowanego wyjścia (dekompozycja zapytania, dobór tagów). Dla Groq używa `response_format={"type": "json_object"}` — API gwarantuje poprawny JSON. Dla Ollama Cloud używa parametru `"format": "json"`. W obu przypadkach system prompt zawiera instrukcję "Odpowiadaj WYŁĄCZNIE poprawnym JSON. Bez komentarzy."

---

## 8. Budowanie kontekstu (ui.py → build_context)

Kontekst to tekst przekazywany do LLM jako "dokumenty". Jego jakość bezpośrednio wpływa na jakość odpowiedzi.

### 8.1 Sortowanie dokumentów

Przed budowaniem kontekstu dokumenty są sortowane według priorytetu:

```python
_CONTEXT_TYPE_ORDER = {
    "uodo_decision":     0,   # pierwsze
    "legal_act_article": 1,
    "gdpr_article":      2,
    "gdpr_recital":      3,   # ostatnie
}
docs_sorted = sorted(docs, key=lambda d: (
    _CONTEXT_TYPE_ORDER.get(d.get("doc_type"), 9),
    -d.get("_score", 0)  # w ramach typu: wyższy score = wyżej
))
```

**Dlaczego to ważne?** Duże modele językowe mają tendencję do "zapominania" informacji z środka długiego kontekstu (tzw. "lost in the middle"). Umieszczenie decyzji UODO na początku zapewnia, że model widzi je zanim skończy się limit tokenów.

### 8.2 Szablony Jinja2

Każdy typ dokumentu ma własny szablon, co pozwala na precyzyjne "zakotwiczenie uwagi" modelu — LLM widzi wyraźne etykiety (`DECYZJA UODO`, `USTAWA`, `RODO`) i nie musi sam kategoryzować dokumentów:

```
[1] DECYZJA UODO DKN.5110.16.2022 (2022-08, prawomocna)
  SYGNATURA:     DKN.5110.16.2022
  DATA:          2022-08
  STATUS:        prawomocna
  TAGI:          dane genetyczne, zdrowie, brak podstawy prawnej
  POWOŁANE AKTY: Dz.U. 2019 poz. 1781, EU 2016/679
  TREŚĆ:
  [fragment treści]

---
[2] USTAWA o ochronie danych osobowych — Art. 9
  ŹRÓDŁO: Dz.U. 2019 poz. 1781 (u.o.d.o.)
  TREŚĆ:
  [treść artykułu]

---
[3] RODO (rozporządzenie 2016/679) — Art. 9
  ŹRÓDŁO: Dz.Urz. UE L 119/1
  TREŚĆ:
  [treść artykułu]
```

Nagłówek kontekstu jawnie informuje model o wszystkich trzech typach dokumentów — wcześniej brak tej informacji powodował, że duże modele (jak kimi2.5) interpretowały kontekst jako "wyłącznie przepisy RODO" i twierdziły, że nie ma decyzji UODO.

### 8.3 Ekstrakcja fragmentów

Decyzje UODO mogą mieć do 50 000 znaków. Do kontekstu trafia max 2000 znaków — `_extract_fragment()` wybiera okno o najwyższej gęstości słów kluczowych z zapytania:

```python
# Przeszukuje treść krokiem 150 znaków
# Dla każdej pozycji liczy, ile razy słowa kluczowe pojawiają się w oknie 2000 znaków
# Wybiera pozycję z najwyższym wynikiem
step = 150
for pos in range(0, len(content) - max_len, step):
    score = sum(content[pos:pos+max_len].count(kw) for kw in keywords)
    if score > best_score:
        best_score, best_pos = score, pos
```

### 8.4 Limit znaków

Kontekst jest budowany do limitu 18 000 znaków. Gdy kolejny blok przekroczyłby limit, pętla się przerywa i dodaje notatkę `[pominięto N dalszych wyników]`. Ponieważ decyzje są sortowane jako pierwsze, mają gwarancję trafienia do kontekstu przed artykułami RODO.

---

## 9. Interfejs użytkownika (main.py + ui.py)

### 9.1 Streamlit i session_state

Streamlit rerenderuje cały skrypt przy każdej interakcji użytkownika. `st.session_state` to słownik persystujący między rerenderami — przechowuje m.in.:
- `llm_provider`, `llm_model`, `llm_api_key` — konfiguracja LLM z sidebara
- `last_query`, `last_filters` — żeby nie wykonywać wyszukiwania przy każdym rerenderze
- `agent_memory` — obiekt `AgentMemory` z historią sesji
- `_example_query` — tymczasowe przechowywanie pytania klikniętego z przykładów

### 9.2 Trigger wyszukiwania

Wyszukiwanie uruchamia się gdy:
```python
if effective_query and (
    search_btn                                           # kliknięto "Szukaj"
    or st.session_state.get("last_query") != query       # zmieniło się zapytanie
    or st.session_state.get("last_filters") != str(filters)  # zmieniły się filtry
):
```

Ta logika zapobiega wielokrotnemu wyszukiwaniu przy każdym rerenderze Streamlit.

### 9.3 Fast path po sygnaturze

Jeśli zapytanie pasuje do wyrażenia regularnego sygnatury (`DKN.XXXX.XX.XXXX`), aplikacja pomija wyszukiwanie semantyczne i bezpośrednio pobiera decyzję z Qdrant przez `fetch_by_signature()`. Ewentualnie dokłada powiązane decyzje z grafu cytowań.

### 9.4 Zakładki wyników

Wyniki są prezentowane w pięciu zakładkach:
- **Wszystkie** — wszystkie dokumenty w kolejności zwróconej przez wyszukiwanie
- **Decyzje UODO** — tylko orzeczenia, z pełnymi kartami
- **Ustawa u.o.d.o.** — artykuły ustawy
- **RODO** — artykuły i motywy RODO
- **Graf** — decyzje dodane przez rozszerzenie grafu (oznaczone typem relacji)

### 9.5 Karty dokumentów

Każdy typ dokumentu ma własną funkcję renderowania (`render_decision_card`, `render_act_article_card`, `render_gdpr_card`). Karty decyzji wyświetlają:
- Sygnaturę jako link do portalu UODO
- Status prawny (kolorowy badge: zielony = prawomocna, niebieski = nieprawomocna)
- Tytuł pełny (opis naruszenia)
- Tagi (pierwszych 8, z informacją ile więcej)
- Powołane akty prawne

CSS jest wzorowany bezpośrednio na portalu orzeczenia.uodo.gov.pl (zmienne CSS, typografia Red Hat Display).

---

## 10. Pamięć epizodyczna

`AgentMemory` przechowuje ostatnie 5 wyszukiwań z bieżącej sesji. Każdy wpis (`MemoryEntry`) zawiera:
- Oryginalne i wzbogacone zapytanie
- Streszczenie dekompozycji (reasoning LLM)
- Sygnatury znalezionych decyzji (top 5)
- Numery artykułów u.o.d.o. (top 3)
- Pierwszych 300 znaków odpowiedzi AI

### Zastosowanie w kontekście

Jeśli bieżące zapytanie ma wspólne słowa z poprzednim zapytaniem, `find_related()` dołącza do kontekstu LLM notatkę:

```
KONTEKST Z POPRZEDNICH ANALIZ (tej sesji):
- Poprzednie pytanie: «dane genetyczne» → znalezione decyzje: DKN.5110.16.2022, DKN.5131.9.2025
```

Pozwala to LLM uwzględnić poprzednie wyniki bez konieczności ponownego wyszukiwania.

### Zastosowanie w sidbarze

Historia sesji jest widoczna w panelu bocznym jako ekspandery — użytkownik widzi jakie pytania zadał, które decyzje znalazł i fragment odpowiedzi AI.

---

## 11. Konfiguracja i modele danych

### 11.1 config.py

Centralne miejsce wszystkich stałych. Kluczowe wartości:

```python
MAX_ACT_DOCS  = 5    # max artykułów u.o.d.o. w wynikach
MAX_GDPR_DOCS = 3    # max artykułów RODO w wynikach
TOP_K         = 8    # domyślne top_k dla semantic search
GRAPH_DEPTH   = 2    # głębokość przeszukiwania grafu
```

`QUERY_STOPWORDS` — zbiór polskich słów funkcyjnych pomijanych przy ekstrakcji fraz z zapytania. Bez nich "w jakie dane genetyczne są przetwarzane" generowałby frazy "jakie dane", "dane genetyczne" — stopwords zostawia tylko "dane genetyczne".

### 11.2 models.py — modele Pydantic

`QueryDecomposition` i `AgentMemory` / `MemoryEntry` są modelami Pydantic — zapewnia to walidację typów i czytelną serializację. Jeśli LLM zwróci niepoprawny JSON, Pydantic rzuca wyjątek który jest obsługiwany z fallbackiem.

Szablony Jinja2 (`TPL_HEADER`, `TPL_DECISION` itd.) są kompilowane raz przy imporcie modułu i współdzielone przez cały czas życia aplikacji.

---

## 12. Narzędzia pomocnicze (tools/)

### eval.py — ewaluacja jakości

10 złotych pytań z binarnymi kryteriami sprawdzenia. Każde pytanie ma 3 check funkcje (lambda) testujące obecność konkretnych słów/sygnatur/artykułów w odpowiedzi LLM. Wynik: `passed/total` dla każdego pytania + agregat procentowy. Wyniki zapisywane do `eval_results.json`.

Przykład:
```python
{
    "question": "Kiedy wymagane jest zgłoszenie naruszenia danych do UODO?",
    "checks": [
        lambda a: "72" in a,                           # podaje termin 72h
        lambda a: "art" in a.lower() and "33" in a,   # cytuje Art. 33 RODO
        lambda a: "naruszenie" in a.lower(),           # używa właściwego pojęcia
    ]
}
```

### enrich_act_keywords.py

Dla artykułów u.o.d.o. i RODO które nie mają tagów, wywołuje LLM z treścią artykułu i listą istniejących tagów z decyzji UODO. LLM wybiera pasujące tagi i zapisuje je bezpośrednio do Qdrant przez `client.set_payload()` — bez przeindeksowania.

---

## 13. Kluczowe decyzje projektowe

### Dlaczego jedna kolekcja Qdrant?

Wszystkie trzy typy dokumentów (decyzje, ustawa, RODO) trafiają do jednej kolekcji `uodo_decisions`. Alternatywą byłyby osobne kolekcje, ale wymuszałoby to osobne zapytania i ręczne łączenie wyników. Jedna kolekcja pozwala filtrować po `doc_type` w jednym zapytaniu i utrzymuje prostszą architekturę.

### Dlaczego frazy z zapytania zamiast tylko LLM do tagów?

Pierwsze podejście używało LLM do doboru tagów zawsze. Problem: dla zapytania "dane genetyczne" LLM proponował też "dane biometryczne", "dane szczególnych kategorii", "zdrowie" — co dawało 100+ wyników zamiast 26. Bezpośrednie dopasowanie fraz jest deterministyczne, szybkie (cache tagów) i precyzyjne.

### Dlaczego nie przebudować embeddingów i zrobić full-text search zamiast tagów?

Semantyczny search rozumie sens, ale nie gwarantuje pełnego recall. Dla pytania prawnego "jakie decyzje dotyczą danych genetycznych" użytkownik oczekuje **wszystkich** decyzji z tagiem — nie top-k najbardziej semantycznie podobnych. Tagi są precyzyjne, bo zostały przypisane przez ekspertów portalu UODO, a nie wygenerowane automatycznie.

### Dlaczego osobne buckety zamiast jednej listy wyników?

Gdyby wszystkie dokumenty były w jednej liście sortowanej po score, artykuły RODO (które mają wysoki score semantyczny dla pytań o dane genetyczne, bo Art. 9 RODO definiuje je wprost) wypychałyby decyzje UODO poza limit kontekstu. Osobne buckety z twardymi limitami (max 5 u.o.d.o., max 3 RODO) gwarantują, że decyzje UODO zawsze trafiają do kontekstu LLM.

### Dlaczego Streamlit zamiast FastAPI + React?

Streamlit pozwala na bardzo szybkie prototypowanie i iterowanie bez osobnego frontendu. Dla wewnętrznego narzędzia analitycznego jest wystarczający. Wadą jest ograniczona kontrola nad UI i rerenderowanie przy każdej interakcji — stąd `st.session_state` do cache'owania wyników wyszukiwania.

### Dlaczego graf zapisywany do pliku .pkl?

Budowanie grafu wymaga scrollowania przez całą kolekcję Qdrant (~560 decyzji × metadane). Przy każdym starcie aplikacji byłoby to powolne. Plik `.pkl` jest ładowany w ułamku sekundy. Graf jest przebudowywany tylko gdy plik nie istnieje — po dodaniu nowych decyzji trzeba go usunąć ręcznie, żeby wymusić przebudowanie.
