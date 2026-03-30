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

### Co to jest RAG?

Aplikacja jest systemem **RAG (Retrieval-Augmented Generation)** — dosłownie "Generowanie wspomagane wyszukiwaniem". Żeby zrozumieć po co to istnieje, warto najpierw zrozumieć problem z samymi modelami językowymi.

Modele AI jak GPT czy Claude są trenowane na ogromnych zbiorach tekstu z internetu. Wiedzą dużo o świecie, ale ich wiedza ma dwie fundamentalne wady:
1. **Jest zamrożona w czasie** — model nie wie co wydarzyło się po dacie zakończenia treningu
2. **Może być nieprecyzyjna lub zmyślona** — modele potrafią generować przekonująco brzmiące, ale fałszywe informacje (tzw. halucynacje)

RAG rozwiązuje oba problemy: zamiast polegać na wiedzy modelu, **najpierw wyszukujemy odpowiednie dokumenty** z naszej własnej bazy, a potem podajemy je modelowi jako materiał źródłowy. Model odpowiada wyłącznie na podstawie tego co dostał — jak prawnik który cytuje konkretne przepisy, a nie swoją ogólną wiedzę o prawie.

### Problem który rozwiązuje

Baza decyzji Prezesa UODO liczy ponad 560 orzeczeń administracyjnych. Każde orzeczenie to kilka do kilkudziesięciu stron tekstu prawniczego. Analityk szukający precedensów dla konkretnego problemu (np. przetwarzania danych biometrycznych przez pracodawcę) musiałby ręcznie przejrzeć setki dokumentów.

Tradycyjne wyszukiwanie pełnotekstowe (jak w Google) szuka dokładnych słów — nie rozumie sensu pytania. Zapytanie "jak firma powinna postąpić gdy pracownik odmawia zgody na monitoring" nie znajdzie decyzji opisującej "brak podstawy prawnej przetwarzania w stosunku pracy", choć semantycznie mówią o tym samym.

Aplikacja łączy trzy techniki: wyszukiwanie po tagach (precyzyjne), wyszukiwanie semantyczne (rozumie sens) i syntezę odpowiedzi przez LLM (formułuje czytelną odpowiedź z odniesieniami do źródeł).

### Trzy źródła wiedzy

- **Decyzje UODO** — ~560 orzeczeń administracyjnych Prezesa Urzędu Ochrony Danych Osobowych, pobieranych z portalu orzeczenia.uodo.gov.pl
- **Ustawa o ochronie danych osobowych (u.o.d.o.)** — artykuły 1–108, Dz.U. 2019 poz. 1781
- **RODO** — 99 artykułów i 173 motywy preambuły rozporządzenia (UE) 2016/679

---

## 2. Architektura ogólna

Aplikacja składa się z sześciu modułów Python oraz zestawu narzędzi w katalogu `tools/`:

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

- **Qdrant** — wektorowa baza danych. Przechowuje dwa rodzaje danych dla każdego dokumentu: matematyczną reprezentację jego treści (embedding/wektor) oraz metadane (tagi, sygnatura, data, status). Działa jako osobny serwis Docker.
- **SentenceTransformers** — biblioteka do generowania embeddingów. Używa polskiego modelu `sdadas/mmlw-retrieval-roberta-large` uruchamianego lokalnie na CPU.
- **NetworkX** — biblioteka do grafu powiązań między decyzjami. Graf cytowań pozwala znaleźć decyzje powiązane tematycznie nawet jeśli nie zawierają szukanej frazy.
- **Ollama** — lokalna instalacja daemona LLM. Modele cloud (np. `gpt-oss:120b-cloud`) pobierają tokeny z serwerów Ollama.com ale przez lokalny punkt dostępowy na porcie 11434.
- **Groq** — alternatywne zewnętrzne API dla modeli LLM, z bezpłatnym limitem.
- **Streamlit** — framework webowy dla interfejsu użytkownika. Zamienia skrypt Python w interaktywną aplikację webową bez konieczności pisania HTML/JavaScript.

---

## 3. Baza wiedzy i indeksowanie

Zanim aplikacja zacznie działać, dokumenty muszą zostać przetworzone i zaindeksowane w Qdrant. To jednorazowy (lub rzadki) proces wykonywany przez narzędzia z katalogu `tools/`.

### 3.1 Co to jest embedding i dlaczego jest potrzebny?

Komputery nie rozumieją tekstu — operują na liczbach. **Embedding** to sposób zamiany tekstu na listę liczb (wektor) w taki sposób, że teksty o podobnym znaczeniu mają podobne wektory, a teksty o różnym znaczeniu — różne.

Przykład: zdania "przetwarzanie danych osobowych bez zgody" i "brak podstawy prawnej przetwarzania" są napisane różnymi słowami, ale opisują to samo zjawisko prawne. Model embeddingowy umieszcza je blisko siebie w przestrzeni matematycznej. Zdanie "przepis na bigos" trafia daleko od obu.

Wektory mają wymiar ~1024 liczb — każda liczba reprezentuje pewien abstrakcyjny wymiar semantyczny. Nie da się tego intuicyjnie zinterpretować, ale matematyczna odległość między wektorami jest precyzyjną miarą podobieństwa semantycznego.

### 3.2 Struktura kolekcji Qdrant

Wszystkie dokumenty trafiają do **jednej kolekcji** o nazwie `uodo_decisions`. Każdy punkt (dokument) w Qdrant ma:

- **Wektor** — 1024 liczb reprezentujących semantyczne znaczenie treści
- **Payload** — słownik metadanych dostępny do filtrowania i wyświetlania

Kluczowe pola payloadu:

| Pole | Typ | Przykład |
|------|-----|---------|
| `doc_type` | keyword | `uodo_decision` |
| `signature` | keyword | `DKN.5110.16.2022` |
| `keywords` | keyword[] | `["dane genetyczne", "dane szczególnych kategorii", "zdrowie"]` |
| `status` | keyword | `prawomocna` |
| `year` | integer | `2022` |
| `content_text` | text | Pełna treść decyzji (do 50 000 znaków) |
| `term_decision_type` | keyword[] | `["nakaz"]` |
| `term_violation_type` | keyword[] | `["brak podstawy prawnej przetwarzania"]` |
| `term_legal_basis` | keyword[] | `["zgoda osoby, której dane dotyczą"]` |
| `term_corrective_measure` | keyword[] | `["administracyjna kara pieniężna"]` |
| `term_sector` | keyword[] | `["Zdrowie"]` |
| `related_uodo_rulings` | text[] | `["DKN.5131.9.2021", "ZSOŚS.440.21.2019"]` |
| `related_acts` | text[] | `["Dz.U. 2019 poz. 1781"]` |
| `related_eu_acts` | text[] | `["EU 2016/679"]` |

### 3.3 Indeksowanie decyzji UODO

**Etap 1 — Scraping:** `uodo_scraper.py` pobiera decyzje przez REST API portalu `orzeczenia.uodo.gov.pl`. Dla każdej decyzji pobiera treść pełną, metadane (tytuł, tagi, podmioty, rodzaj decyzji), daty i powiązania z innymi aktami.

Powiązania z innymi aktami są wyciągane dwutorowo: z API (`refs` w meta.json) oraz fallbackiem przez wyrażenia regularne bezpośrednio z treści decyzji — wzorce dla `Dz.U. XXXX poz. XXXX`, `EU 2016/679`, sygnatur UODO i wyroków NSA/WSA.

**Etap 2 — Wzbogacenie taksonomii:** `enrich_jsonl_taxonomy.py` mapuje numerowane etykiety z API (np. `1.1`, `2.3`, `9.1`) na konkretne pola semantyczne. Prefix numeru determinuje kategorię:
- `1.x` → `term_decision_type` (rodzaj decyzji)
- `2.x` → `term_violation_type` (rodzaj naruszenia)
- `9.x` → `term_sector` (sektor)

**Etap 3 — Indeksowanie:** `uodo_indexer.py` buduje tekst do embeddingu — to nie jest surowa treść decyzji, ale jej skondensowana reprezentacja:

```
DKN.5110.16.2022 Przetwarzanie danych zdrowotnych bez podstawy prawnej prawomocna
Słowa kluczowe: dane genetyczne, zdrowie, brak podstawy prawnej, Art. 9 RODO
Podmiot: Szpital Miejski w Krakowie

[pierwsze 5500 znaków treści decyzji]

Akty: Dz.U. 2019 poz. 1781 EU 2016/679
```

Sygnatura i tytuł na początku mają najwyższą wagę przy wyszukiwaniu semantycznym. UUID punktu jest deterministyczny — wyznaczany przez MD5 sygnatury, więc ponowne indeksowanie tej samej decyzji nadpisuje poprzedni punkt zamiast tworzyć duplikat.

### 3.4 Indeksowanie ustawy i RODO

**Ustawa u.o.d.o.** — parser rozpoznaje nagłówki artykułów (`Art. X.`) w pliku Markdown i wyodrębnia ich treść. Artykuły dłuższe niż 3000 znaków są dzielone na mniejsze fragmenty (chunki) z overlapem 300 znaków — overlap zapobiega urwaniu kontekstu na granicy fragmentów.

**RODO** — parser rozróżnia motywy preambuły (`- (N) treść`) i artykuły (`# Artykuł N`). Motywy nie są dzielone (są krótkie i stanowią zamkniętą całość). Artykuły dłuższe niż 1200 znaków są chunkowane z overlapem 100 znaków.

---

## 4. Przepływ danych — od zapytania do odpowiedzi

To jest najważniejsza sekcja — opisuje dokładnie co dzieje się od momentu gdy użytkownik wpisuje pytanie do momentu gdy widzi odpowiedź. Poniżej skrócony schemat przepływu, a następnie szczegółowy opis każdego kroku na dwóch przykładach:

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

Prześledzę cały proces na dwóch przykładach:

- **Przykład A**: krótkie, precyzyjne zapytanie — `"Dane genetyczne"`
- **Przykład B**: długie, opisowe zapytanie — `"Jakie obowiązki ma pracodawca przy monitoringu wizyjnym pracowników?"`

---

### Krok 0 — Użytkownik wpisuje zapytanie

Użytkownik wpisuje tekst w polu wyszukiwania i klika "Szukaj".

Aplikacja sprawdza najpierw czy zapytanie wygląda jak sygnatura decyzji (np. `DKN.5110.16.2022`) — jeśli tak, pomija całą logikę wyszukiwania i bezpośrednio pobiera tę decyzję z Qdrant. To tzw. **fast path**.

Dla obu naszych przykładów zapytania nie są sygnaturami, więc przechodzimy do normalnego przepływu.

---

### Krok 1 — Reasoning Step (dekompozycja zapytania)

**Co to jest i po co?**

Zanim zaczniemy szukać dokumentów, LLM analizuje zapytanie i wyciąga z niego ustrukturyzowane informacje. To jak briefing przed wyszukiwaniem — zamiast iść do biblioteki z kartką "dane genetyczne", idziemy z pełną listą: jakich dokumentów szukamy, które artykuły mogą być istotne, z jakiego okresu.

Ten krok jest uruchamiany **tylko dla zapytań dłuższych niż 3 słowa** i tylko gdy AI jest włączone.

**Przykład A (`"Dane genetyczne"`)** — 2 słowa, krok pominięty. Zapytanie trafia dalej bez modyfikacji.

**Przykład B (`"Jakie obowiązki ma pracodawca przy monitoringu wizyjnym pracowników?"`)** — 8 słów, krok uruchamiany.

LLM otrzymuje zapytanie i zwraca JSON z analizą:

```json
{
  "query_type": "szukam_decyzji",
  "search_keywords": [
    "monitoring wizyjny",
    "monitoring w miejscu pracy",
    "kamera w pracy",
    "nadzór nad pracownikami"
  ],
  "gdpr_articles_hint": ["Art. 5", "Art. 6", "Art. 9", "Art. 88"],
  "uodo_act_articles_hint": ["Art. 22²"],
  "year_from_hint": null,
  "year_to_hint": null,
  "enriched_query": "monitoring wizyjny kamera pracodawca pracownicy obowiązki informacyjne podstawa prawna",
  "reasoning": "Pytanie dotyczy monitoringu wizyjnego w miejscu pracy — kluczowe są Kodeks pracy Art. 22² oraz RODO Art. 88."
}
```

Z tego wyniku aplikacja wyciąga:
- `search_keywords[:3]` → `"monitoring wizyjny monitoring w miejscu pracy kamera w pracy"` — to zostanie użyte do wyszukiwania semantycznego
- `gdpr_articles_hint` i `uodo_act_articles_hint` → widoczne użytkownikowi w sekcji "Reasoning Step"
- `year_from_hint`/`year_to_hint` → gdyby były ustawione, zawęziłyby wyniki do danego okresu

**Dlaczego `search_keywords` zamiast `enriched_query`?**

`enriched_query` to często długie zdanie opisowe, które przy zamianie na wektor "rozmywa" znaczenie — embedding próbuje reprezentować zbyt wiele różnych pojęć naraz. Kilka konkretnych fraz kluczowych (`"monitoring wizyjny monitoring w miejscu pracy"`) daje bardziej skupiony wektor i trafniejsze wyniki semantyczne.

---

### Krok 2 — Hybrid Search (wyszukiwanie hybrydowe)

**Co to jest?**

"Hybrid" oznacza że łączymy dwie techniki: wyszukiwanie po tagach (dokładne) i wyszukiwanie semantyczne (rozumie sens). Wyniki są zbierane do trzech osobnych "kubełków" — decyzji UODO, artykułów u.o.d.o. i artykułów RODO — z różnymi limitami dla każdego.

**Skąd bierze się oryginalne vs. wzbogacone zapytanie?**

Do wyszukiwania po tagach zawsze używamy **oryginalnego zapytania** użytkownika — frazy "monitoring wizyjny" które on wpisał, nie te wymyślone przez LLM. Do wyszukiwania semantycznego używamy **`search_keywords`** z dekompozycji — bardziej skupionego zestawu pojęć.

#### Kubełek 1 — Decyzje UODO (bez limitu liczby)

Wypełniany w czterech krokach z malejącym priorytetem:

**Krok 1a — explicit keyword z UI:**
Użytkownik może wybrać konkretny tag z listy w sekcji "Filtry zaawansowane". Jeśli wybrał np. `"monitoring wizyjny"`, aplikacja pobiera z Qdrant **wszystkie** decyzje z tym tagiem bez żadnego limitu — może to być 15, 50 lub 200 dokumentów.

Dla obu naszych przykładów zakładamy że użytkownik nie ustawił ręcznie tagu, więc krok 1a jest pominięty.

**Krok 1b — frazy z zapytania (BEZ LLM):**

To kluczowy krok. Aplikacja wyciąga z oryginalnego zapytania wszystkie możliwe frazy 2-wyrazowe i sprawdza czy któraś z nich jest tagiem w bazie.

```
Przykład A: query = "Dane genetyczne"
  words = ["dane", "genetyczne"]
  frazy 2-wyrazowe = ["dane genetyczne"]
  
  Sprawdzamy czy "dane genetyczne" jest tagiem w bazie → TAK
  → keyword_exact_search("dane genetyczne") → 26 decyzji
  
  Krok 1b zakończony sukcesem. Kroki 1c i 1d pominięte.

Przykład B: query = "Jakie obowiązki ma pracodawca przy monitoringu wizyjnym pracowników?"
  words = ["jakie", "obowiązki", "pracodawca", "przy", "monitoringu", "wizyjnym", "pracowników"]
  → po usunięciu stopwords: ["obowiązki", "pracodawca", "monitoringu", "wizyjnym", "pracowników"]
  frazy 2-wyrazowe = ["obowiązki pracodawca", "pracodawca monitoringu", 
                      "monitoringu wizyjnym", "wizyjnym pracowników"]
  
  Sprawdzamy każdą:
  "obowiązki pracodawca" → nie jest tagiem
  "pracodawca monitoringu" → nie jest tagiem
  "monitoringu wizyjnym" → nie jest tagiem (tag to "monitoring wizyjny" bez odmiany!)
  "wizyjnym pracowników" → nie jest tagiem
  
  Krok 1b: zero trafień → przechodzimy do 1c
```

Uwaga na odmianę — to znane ograniczenie systemu tagów. Tag w bazie to `"monitoring wizyjny"` (mianownik), ale w zapytaniu pojawia się `"monitoringu wizyjnym"` (odmiana). Porównanie działa na poziomie tekstu, nie lemmatyzacji. Dlatego istnieje krok 1c z LLM.

**Krok 1c — tagi LLM (fallback):**

Uruchamiany tylko gdy krok 1b nic nie znalazł. LLM dostaje pełną listę tagów z bazy i wybiera pasujące do zapytania. Może obsługiwać synonimy i odmiany.

```
Przykład B — LLM dostaje listę tagów i zapytanie:
  → proponuje: ["monitoring wizyjny", "monitoring w miejscu pracy", 
                "obowiązek informacyjny", "zatrudnienie"]
  
  Filtrujemy — odrzucamy tagi zwracające >50 decyzji (zbyt ogólne):
  "monitoring wizyjny" → np. 23 decyzje → OK → keyword_exact_search → 23 decyzje
  "monitoring w miejscu pracy" → np. 8 decyzji → OK → +8 decyzji (deduplikacja)
  "obowiązek informacyjny" → np. 180 decyzji → ZBYT OGÓLNY → pomijamy
  "zatrudnienie" → to tag taksonomii sektora, nie słowo kluczowe → pomijamy

  Wynik: ~25-30 decyzji o monitoringu
```

Limit 50 decyzji per tag jest ochroną przed zaśmieceniem wyników — tagi jak "dane osobowe" czy "zgoda" pasują do większości bazy i są bezużyteczne jako filtr.

**Krok 1d — semantic search (ostatni fallback):**

Uruchamiany tylko gdy mamy mniej niż 5 decyzji po krokach 1a-1c. Używa wektora z `search_keywords` i szuka podobieństwa kosinusowego w przestrzeni embeddingów.

```
Przykład: query o bardzo niszowym temacie bez tagu w bazie
  "Czy inspektor sanitarny może żądać danych osobowych pacjentów?"
  
  Krok 1b: brak dwuwyrazowych fraz będących tagiem
  Krok 1c: LLM proponuje tagi, ale żaden nie zwraca wyników
  Krok 1d: semantic_search("inspektor sanitarny dane osobowe pacjentów", top_k=20, threshold=0.45)
  
  Qdrant oblicza odległość kosinusową między wektorem zapytania
  a wektorami wszystkich ~560 decyzji.
  Zwraca 20 najbardziej podobnych semantycznie.
```

Próg `0.45` jest wysoki (skala 0-1) — oznacza że zwracamy tylko decyzje o silnym podobieństwie semantycznym. Pozwala to uniknąć zupełnie niepowiązanych wyników.

#### Kubełek 2 — Artykuły u.o.d.o. (max 5)

Dla obu przykładów: semantic search z `search_keywords` w typie `legal_act_article`, próg `0.25`.

```
Przykład A: search_keywords = "dane genetyczne" (brak dekompozycji)
  → Art. 14 u.o.d.o. (dane szczególnych kategorii) — score 0.71
  → Art. 9 u.o.d.o. (przetwarzanie szczególnych kategorii) — score 0.68
  → Art. 88 u.o.d.o. (przetwarzanie w zatrudnieniu) — score 0.31

Przykład B: search_keywords = "monitoring wizyjny monitoring w miejscu pracy kamera w pracy"
  → Art. 22² u.o.d.o. (monitoring w zakładzie pracy) — score 0.89
  → Art. 12 u.o.d.o. (obowiązki informacyjne) — score 0.54
  → Art. 9 u.o.d.o. (dane szczególnych kategorii) — score 0.28
```

#### Kubełek 3 — Artykuły RODO (max 3)

Analogicznie, próg `0.30` (wyższy = bardziej rygorystyczny = mniej wyników).

```
Przykład A:
  → Art. 4 ust. 13 RODO (definicja danych genetycznych) — score 0.82
  → Art. 9 RODO (przetwarzanie szczególnych kategorii) — score 0.79
  → Motyw 34 RODO (dane genetyczne a zdrowie) — score 0.61

Przykład B:
  → Art. 88 RODO (przetwarzanie w kontekście zatrudnienia) — score 0.77
  → Art. 5 RODO (zasady przetwarzania) — score 0.52
  → Art. 13 RODO (obowiązek informacyjny) — score 0.48
```

#### Graf cytowań — rozszerzenie wyników

Po zebraniu decyzji z kubełka 1, aplikacja używa grafu cytowań żeby dodać powiązane decyzje. Graf to sieć cytowań między decyzjami — jeśli decyzja A cytuje decyzję B, między nimi jest krawędź.

```
Przykład A — znaleziono 26 decyzji z tagiem "dane genetyczne":
  Seed sigs = [DKN.5110.16.2022, DKN.5131.9.2021, ...]
  
  graph_expand() sprawdza dla każdej znalezionej decyzji:
  - Które decyzje ona cytuje? (następniki) → dodaj z wagą 0.6
  - Które decyzje ją cytują? (poprzedniki) → dodaj z wagą 0.5
  
  Wynik: np. DKN.5132.4.2023 nie ma tagu "dane genetyczne",
  ale cytuje 3 decyzje z naszej listy → trafia do zakładki "Graf"
  z informacją "cytuje tę decyzję"
```

Waga `0.6` i `0.5` (i jej spadek o `0.65` przy każdym kolejnym poziomie głębokości) służy do sortowania — decyzje bezpośrednio powiązane są wyżej niż te powiązane przez pośrednika.

**Wynik końcowy wyszukiwania:**

```
Przykład A ("Dane genetyczne"):
  → 26 decyzji z tagu "dane genetyczne"
  → +3 z grafu (cytują decyzje z listy)
  → +3 artykuły u.o.d.o.
  → +3 artykuły RODO
  Razem: 35 dokumentów, czas: ~2-5 sekund

Przykład B ("Jakie obowiązki ma pracodawca przy monitoringu..."):
  → ~25-30 decyzji z tagów (monitoring wizyjny, monitoring w miejscu pracy)
  → +kilka z grafu
  → +3-5 artykułów u.o.d.o.
  → +3 artykuły RODO
  Razem: ~35-40 dokumentów, czas: ~5-15 sekund (+ czas dekompozycji LLM)
```

---

### Krok 3 — Build Context (budowanie kontekstu dla LLM)

**Co to jest kontekst?**

Kontekst to tekst który zostanie dosłownie wklejony do promptu dla modelu LLM. To jak teczka z dokumentami którą dajemy prawnikowi przed konsultacją — bez niej odpowie z pamięci, z nią odpowie na podstawie konkretnych akt.

**Sortowanie:**

Przed budowaniem kontekstu dokumenty są sortowane — decyzje UODO zawsze pierwsze, RODO ostatnie. To ważne, bo modele językowe mają tendencję do "zapominania" informacji z środka długiego tekstu. Umieszczenie decyzji UODO na początku gwarantuje że model je przetworzy.

**Limit 18 000 znaków:**

Modele mają ograniczone "okno kontekstowe" — maksymalną liczbę znaków którą mogą przetworzyć naraz. Aplikacja buduje kontekst dokument po dokumencie, aż do osiągnięcia limitu 18 000 znaków. Gdy limit jest osiągnięty, dodaje notatkę `[pominięto N dalszych wyników]`.

Dla 26 decyzji o danych genetycznych — decyzje mają średnio 500-2000 znaków fragmentu (fragment, nie pełna treść), więc do limitu wejdzie ~10-20 decyzji plus wszystkie artykuły u.o.d.o. i RODO.

**Ekstrakcja fragmentów:**

Decyzje UODO mogą mieć do 50 000 znaków, ale do kontekstu trafia max 2000 znaków. Algorytm przesuwa okno o 150 znaków po całej treści decyzji i liczy ile razy słowa kluczowe z zapytania pojawiają się w każdym oknie. Wybiera okno z najwyższym wynikiem.

```
Przykład: decyzja DKN.5110.16.2022 (20 000 znaków)
  Zapytanie: "dane genetyczne"
  Słowa kluczowe: ["dane", "genetyczne"]
  
  Okno na pozycji 0-2000:     "dane genetyczne" pojawia się 0 razy → score=0
  Okno na pozycji 150-2150:   "dane genetyczne" pojawia się 0 razy → score=0
  ...
  Okno na pozycji 4500-6500:  "dane genetyczne" pojawia się 4 razy → score=4 ← NAJLEPSZE
  Okno na pozycji 4650-6650:  "dane genetyczne" pojawia się 3 razy → score=3
  
  Wybrany fragment: znaki 4500-6500 (z prefiksem "[…]" jeśli nie zaczyna od początku)
```

**Przykład gotowego kontekstu (uproszczony):**

```
Poniżej znajdują się dokumenty powiązane z pytaniem: «Dane genetyczne»
Zbiór zawiera trzy typy dokumentów:
  1. DECYZJE UODO — decyzje administracyjne Prezesa UODO
  2. ARTYKUŁY u.o.d.o. — przepisy ustawy o ochronie danych osobowych
  3. ARTYKUŁY RODO — przepisy rozporządzenia (UE) 2016/679
Odpowiadaj na podstawie poniższych dokumentów, ze szczególnym uwzględnieniem DECYZJI UODO.

---
[1] DECYZJA UODO DKN.5110.16.2022 (2022-08, prawomocna)
  SYGNATURA:     DKN.5110.16.2022
  DATA:          2022-08
  STATUS:        prawomocna
  TAGI:          dane genetyczne, zdrowie, brak podstawy prawnej
  POWOŁANE AKTY: Dz.U. 2019 poz. 1781, EU 2016/679
  TREŚĆ:
  […]
  Prezes Urzędu stwierdził, że Szpital przetwarzał dane genetyczne pacjentów
  bez ważnej podstawy prawnej. Zgodnie z art. 9 ust. 1 RODO dane genetyczne
  należą do szczególnych kategorii danych osobowych, których przetwarzanie
  jest co do zasady zakazane...

---
[2] DECYZJA UODO DKN.5131.9.2021 (2021-03, prawomocna)
  ...

---
[15] USTAWA o ochronie danych osobowych — Art. 9
  ŹRÓDŁO: Dz.U. 2019 poz. 1781 (u.o.d.o.)
  TREŚĆ:
  Art. 9. Przetwarzanie danych szczególnych kategorii...

---
[16] RODO (rozporządzenie 2016/679) — Art. 9
  ŹRÓDŁO: Dz.Urz. UE L 119/1
  TREŚĆ:
  Artykuł 9 RODO — Przetwarzanie szczególnych kategorii danych osobowych...
```

---

### Krok 4 — LLM Stream (generowanie odpowiedzi)

**Co się dzieje?**

Aplikacja wysyła do modelu LLM dwa elementy:
1. **System prompt** — instrukcja dla modelu (odpowiadaj po polsku, cytuj sygnatury, nie zmyślaj)
2. **User message** — zapytanie użytkownika + cały kontekst z Kroku 3

Model generuje odpowiedź token po tokenie (słowo po słowie) — to właśnie streamowanie. Użytkownik widzi jak tekst pojawia się na ekranie w czasie rzeczywistym, zamiast czekać na gotową odpowiedź.

**Przykład odpowiedzi dla zapytania A ("Dane genetyczne"):**

```
Na podstawie dostarczonych decyzji UODO można wskazać następujące kluczowe 
ustalenia dotyczące przetwarzania danych genetycznych:

**Definicja i podstawa prawna**
Dane genetyczne są szczególną kategorią danych osobowych w rozumieniu 
Art. 9 ust. 1 RODO i Art. 4 pkt 13 RODO. Ich przetwarzanie jest co do zasady 
zakazane, chyba że zachodzi jedna z przesłanek z Art. 9 ust. 2 RODO.

**Decyzje w sprawach placówek medycznych**
W decyzji DKN.5110.16.2022 Prezes UODO stwierdził naruszenie przez szpital 
przepisów o ochronie danych genetycznych pacjentów — administrator nie posiadał 
ważnej podstawy prawnej i nałożył karę pieniężną...

**Decyzje w sprawach badań naukowych**
Decyzja DKN.5131.9.2021 dotyczyła przetwarzania danych genetycznych w ramach 
badań naukowych — Prezes UODO wskazał że zgoda uczestnika musi być...
```

LLM cytuje konkretne sygnatury i artykuły bo system prompt go do tego zobowiązuje, a kontekst dostarcza mu gotowe informacje do zacytowania.

---

### Krok 5 — Wyświetlenie w Streamlit

Odpowiedź AI pojawia się w niebieskiej ramce na górze. Poniżej — zakładki z dokumentami które zostały znalezione i użyte jako podstawa odpowiedzi. Użytkownik może kliknąć w sygnaturę decyzji żeby przejść bezpośrednio do pełnej treści na portalu UODO.

---

### Krok 6 — Zapis do pamięci epizodycznej

Zapytanie, lista znalezionych decyzji i fragment odpowiedzi są zapisywane w pamięci sesji. Przy kolejnym podobnym zapytaniu LLM dostanie dodatkową notatkę "w poprzednim pytaniu o dane genetyczne znalezione zostały decyzje: DKN.5110.16.2022..." — to pozwala na kontynuację rozmowy bez ponownego wyszukiwania.

---

## 5. Moduł wyszukiwania (search.py)

To serce aplikacji. Funkcja `hybrid_search()` realizuje wieloetapową strategię wyszukiwania z wyraźnymi priorytetami.

### 5.1 Semantic search

`semantic_search()` przyjmuje zapytanie, generuje jego embedding i pyta Qdrant o najbliższe wektory. Qdrant używa algorytmu ANN (Approximate Nearest Neighbor) — zamiast porównywać zapytanie ze wszystkimi 560+ decyzjami po kolei (co byłoby wolne), używa indeksu który pozwala na bardzo szybkie znalezienie najbliższych wektorów.

Parametr `score_threshold` to minimalne podobieństwo kosinusowe które musi mieć dokument żeby trafić do wyników. Skala 0-1 gdzie 1 oznacza identyczne wektory:
- Dla decyzji UODO w fallbacku: `0.45` — rygorystyczny, tylko bardzo podobne
- Dla u.o.d.o.: `0.25` — łagodniejszy, artykuły ustawy są stylistycznie różne od zapytań
- Dla RODO: `0.30` — pośredni

### 5.2 Keyword exact search

`keyword_exact_search()` to scroll bez limitu liczby wyników — pobiera **wszystkie** dokumenty z danym tagiem. To kluczowa różnica względem semantic search — tam jest limit `top_k`, tutaj nie ma. Dzięki temu dla zapytania "dane genetyczne" pobieramy wszystkie 26 decyzji, nie tylko top-8.

Scroll działa stronami po 100 dokumentów — jeśli jest 300 decyzji z tagiem, funkcja wykona 3 zapytania do Qdrant.

### 5.3 Limit per tag

Stała `_MAX_RESULTS_PER_TAG = 50` odrzuca tagi które zwracają zbyt wiele wyników — są zbyt ogólne żeby być użyteczne. Tagi jak "przetwarzanie danych osobowych" (pasuje do 400+ decyzji) lub "zgoda" (200+ decyzji) zaśmiecają wyniki zamiast je filtrować. Tag "dane genetyczne" (26 decyzji) jest precyzyjny i użyteczny.

### 5.4 Deduplikacja

Funkcja `doc_key()` generuje unikalny klucz dla każdego dokumentu. Zbiór `seen_keys` pilnuje żeby ten sam dokument nie trafił do wyników dwukrotnie — decyzja może pasować do kilku tagów jednocześnie, ale powinna pojawić się tylko raz.

---

## 6. Graf powiązań

Graf jest budowany raz (przy pierwszym uruchomieniu) i zapisywany do pliku `uodo_graph.pkl`. Przy kolejnych uruchomieniach jest wczytywany z dysku w ułamku sekundy.

### 6.1 Budowanie grafu

```
G = nx.DiGraph()  # graf skierowany

Węzły = sygnatury decyzji
Krawędzie = cytowania między decyzjami
Typy relacji: CITES_UODO, CITES_ACT, CITES_EU
```

Dla każdej decyzji w Qdrant tworzony jest węzeł, a pole `related_uodo_rulings` zamieniane jest na krawędzie skierowane:

```
DKN.5110.16.2022  ──CITES_UODO──►  DKN.5131.9.2021
DKN.5110.16.2022  ──CITES_ACT───►  Dz.U. 2019 poz. 1781
DKN.5110.16.2022  ──CITES_EU────►  EU 2016/679
```

### 6.2 Rozszerzanie wyników

Po znalezieniu decyzji przez wyszukiwanie, `graph_expand()` przechodzi po grafie w obu kierunkach:
- **Następniki** (decyzje cytowane przez znalezione) — relacja "cytowana", waga 0.6
- **Poprzedniki** (decyzje które cytują znalezione) — relacja "cytuje tę decyzję", waga 0.5

Waga maleje z głębią grafu (`decay = 0.65^d`) — decyzje bezpośrednio powiązane mają wyższy priorytet. Maksymalnie 15 dodatkowych decyzji z grafu, głębokość 2.

Rozszerzenie grafu ma sens prawnie: jeśli znalazłeś decyzję A (precedens), to decyzje które ją cytują prawdopodobnie stosują te same zasady w podobnych sprawach — nawet jeśli używają różnego słownictwa i nie trafią przez wyszukiwanie tagowe.

---

## 7. Moduł LLM (llm.py)

### 7.1 Reasoning Step — dekompozycja zapytania

Przed wyszukiwaniem, dla zapytań dłuższych niż 3 słowa, LLM analizuje pytanie i generuje strukturę `QueryDecomposition` zawierającą: typ zapytania, słowa kluczowe do semantic search, wskazówki artykułów, ewentualne zawężenie dat i uzasadnienie.

Prompt do LLM wymaga odpowiedzi wyłącznie w formacie JSON. Wynik jest parsowany przez Pydantic — jeśli parsowanie się nie powiedzie (model odpowie w złym formacie), używane jest oryginalne zapytanie jako fallback.

### 7.2 Dobór tagów przez LLM

`extract_tags_with_llm()` wysyła do LLM pełną listę tagów z bazy (~400-600 tagów) i prosi o wybór pasujących. LLM może wskazać istniejące tagi (dokładna pisownia) lub zaproponować nowe (z prefiksem `[NOWY]`).

Ta funkcja jest wywoływana zawsze, ale wynik jest używany **tylko jako fallback** (krok 1c) — gdy bezpośrednie dopasowanie fraz z zapytania nic nie znalazło.

### 7.3 Streamowanie odpowiedzi

`call_llm_stream()` generuje tokeny asynchronicznie przez API Ollama (`/api/chat` z `"stream": true`). Każda linia odpowiedzi to JSON z polem `message.content` zawierającym kolejny token. Użytkownik widzi tekst pojawiający się na bieżąco.

System prompt instruuje LLM, żeby: odpowiadał wyłącznie po polsku, zawsze cytował sygnatury decyzji i numery artykułów, wyraźnie poinformował jeśli kontekst nie zawiera odpowiedzi.

### 7.4 JSON bez streamowania

`call_llm_json()` służy do wywołań wymagających strukturyzowanego wyjścia (dekompozycja zapytania, dobór tagów). Używa parametru `"format": "json"` w Ollama. Odpowiedź jest czyszczona z ewentualnych bloków markdown (` ```json ``` `) przez regex, a następnie parsowana.

---

## 8. Budowanie kontekstu (ui.py → build_context)

Kontekst to tekst przekazywany do LLM jako "dokumenty". Jego jakość bezpośrednio wpływa na jakość odpowiedzi.

### 8.1 Sortowanie dokumentów

Przed budowaniem kontekstu dokumenty są sortowane według priorytetu (decyzje UODO pierwsze, RODO ostatnie), a w ramach tego samego typu — malejąco po score wyszukiwania.

Duże modele językowe mają tendencję do "zapominania" informacji z środka długiego kontekstu (tzw. "lost in the middle"). Umieszczenie decyzji UODO na początku gwarantuje że model je przetworzy przed osiągnięciem limitu tokenów.

### 8.2 Szablony Jinja2

Każdy typ dokumentu ma własny szablon z wyraźnymi etykietami. LLM widzi `DECYZJA UODO`, `USTAWA`, `RODO` — nie musi sam kategoryzować dokumentów. Wcześniej brak tej informacji w nagłówku powodował, że duże modele (jak kimi2.5) interpretowały kontekst jako "wyłącznie przepisy RODO" i twierdziły że nie ma decyzji UODO.

### 8.3 Ekstrakcja fragmentów i limit znaków

Decyzje UODO mogą mieć do 50 000 znaków, ale do kontekstu trafia max 2000 znaków — algorytm przesuwa okno o 150 znaków i szuka fragmentu z najwyższą gęstością słów kluczowych.

Kontekst jest budowany do limitu 18 000 znaków. Gdy kolejny blok przekroczyłby limit, pętla się przerywa i dodaje notatkę `[pominięto N dalszych wyników]`.

---

## 9. Interfejs użytkownika (main.py + ui.py)

### 9.1 Streamlit i session_state

Streamlit rerenderuje cały skrypt przy każdej interakcji użytkownika. `st.session_state` to słownik persystujący między rerenderami — przechowuje konfigurację LLM, ostatnie zapytanie i filtry (żeby nie wykonywać wyszukiwania przy każdym rerenderze), historię sesji i ostatnią odpowiedź AI (żeby nie znikała po rerenderze).

### 9.2 Trigger wyszukiwania

Wyszukiwanie uruchamia się gdy kliknięto "Szukaj", zmieniło się zapytanie lub zmieniły się filtry. Ta logika zapobiega wielokrotnemu wyszukiwaniu przy każdym rerenderze Streamlit.

### 9.3 Fast path po sygnaturze

Jeśli zapytanie pasuje do wyrażenia regularnego sygnatury (`DKN.XXXX.XX.XXXX`), aplikacja pomija wyszukiwanie semantyczne i bezpośrednio pobiera decyzję z Qdrant. Ewentualnie dokłada powiązane decyzje z grafu cytowań.

### 9.4 Zakładki wyników

Wyniki są prezentowane w pięciu zakładkach: Wszystkie, Decyzje UODO, Ustawa u.o.d.o., RODO, Graf. Karty decyzji wyświetlają sygnaturę jako link do portalu UODO, status prawny (kolorowy badge), tytuł pełny, tagi i powołane akty.

CSS jest wzorowany bezpośrednio na portalu orzeczenia.uodo.gov.pl — zmienne CSS, typografia Red Hat Display.

---

## 10. Pamięć epizodyczna

`AgentMemory` przechowuje ostatnie 5 wyszukiwań z bieżącej sesji. Każdy wpis zawiera: oryginalne i wzbogacone zapytanie, streszczenie dekompozycji, sygnatury znalezionych decyzji (top 5), numery artykułów u.o.d.o. (top 3) i pierwsze 300 znaków odpowiedzi AI.

Jeśli bieżące zapytanie ma wspólne słowa z poprzednim, do kontekstu LLM dołączana jest notatka:

```
KONTEKST Z POPRZEDNICH ANALIZ (tej sesji):
- Poprzednie pytanie: «dane genetyczne» → znalezione decyzje: DKN.5110.16.2022, DKN.5131.9.2025
```

Pozwala to LLM uwzględnić poprzednie wyniki bez konieczności ponownego wyszukiwania. Historia sesji jest też widoczna w panelu bocznym jako ekspandery.

---

## 11. Konfiguracja i modele danych

### 11.1 config.py

Centralne miejsce wszystkich stałych. Kluczowe wartości:

```python
MAX_ACT_DOCS         = 5    # max artykułów u.o.d.o. w wynikach
MAX_GDPR_DOCS        = 3    # max artykułów RODO w wynikach
TOP_K                = 8    # domyślne top_k dla semantic search
GRAPH_DEPTH          = 2    # głębokość przeszukiwania grafu
_MAX_RESULTS_PER_TAG = 50   # max decyzji per tag (zbyt ogólne tagi odrzucamy)
```

`QUERY_STOPWORDS` — zbiór polskich słów funkcyjnych pomijanych przy ekstrakcji fraz. Bez nich zapytanie "w jakie dane genetyczne są przetwarzane" generowałoby frazy "jakie dane", "dane genetyczne" — stopwords zostawia tylko "dane genetyczne".

### 11.2 models.py — modele Pydantic

`QueryDecomposition` i `AgentMemory`/`MemoryEntry` są modelami Pydantic — gwarantuje to walidację typów i obsługę błędów parsowania JSON od LLM. Szablony Jinja2 są kompilowane raz przy imporcie modułu i współdzielone przez cały czas życia aplikacji.

---

## 12. Narzędzia pomocnicze (tools/)

### eval.py — ewaluacja jakości

10 złotych pytań z binarnymi kryteriami sprawdzenia. Każde pytanie ma 3 funkcje testujące obecność konkretnych słów/sygnatur/artykułów w odpowiedzi LLM. Wynik: `passed/total` dla każdego pytania + agregat procentowy zapisywany do `eval_results.json`.

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

Dla artykułów u.o.d.o. i RODO które nie mają tagów, wywołuje LLM z treścią artykułu i listą istniejących tagów z decyzji UODO. LLM wybiera pasujące tagi i zapisuje je bezpośrednio do Qdrant przez `client.set_payload()` — bez przeindeksowania wektorów.

---

## 13. Kluczowe decyzje projektowe

### Dlaczego jedna kolekcja Qdrant?

Wszystkie trzy typy dokumentów trafiają do jednej kolekcji `uodo_decisions`. Alternatywą byłyby osobne kolekcje, ale wymuszałoby to osobne zapytania i ręczne łączenie wyników. Jedna kolekcja pozwala filtrować po `doc_type` w jednym zapytaniu i utrzymuje prostszą architekturę.

### Dlaczego frazy z zapytania zamiast tylko LLM do tagów?

Pierwsze podejście używało LLM do doboru tagów zawsze. Problem: dla zapytania "dane genetyczne" LLM proponował też "dane biometryczne", "dane szczególnych kategorii", "zdrowie" — co dawało 100+ wyników zamiast 26. Bezpośrednie dopasowanie fraz jest deterministyczne, szybkie (cache tagów) i precyzyjne.

### Dlaczego osobne kubełki zamiast jednej listy wyników?

Gdyby wszystkie dokumenty były w jednej liście sortowanej po score, artykuły RODO (które mają wysoki score semantyczny dla pytań o dane genetyczne, bo Art. 9 RODO definiuje je wprost) wypychałyby decyzje UODO poza limit kontekstu. Osobne kubełki z twardymi limitami (max 5 u.o.d.o., max 3 RODO) gwarantują że decyzje UODO zawsze trafiają do kontekstu LLM.

### Dlaczego Streamlit zamiast FastAPI + React?

Streamlit pozwala na bardzo szybkie prototypowanie i iterowanie bez osobnego frontendu. Dla wewnętrznego narzędzia analitycznego jest wystarczający. Wadą jest ograniczona kontrola nad UI i rerenderowanie przy każdej interakcji — stąd `st.session_state` do cache'owania wyników wyszukiwania i odpowiedzi AI.

### Dlaczego graf zapisywany do pliku .pkl?

Budowanie grafu wymaga scrollowania przez całą kolekcję Qdrant (~560 decyzji × metadane). Przy każdym starcie aplikacji byłoby to powolne. Plik `.pkl` jest ładowany w ułamku sekundy. Graf jest przebudowywany tylko gdy plik nie istnieje — po dodaniu nowych decyzji trzeba go usunąć ręcznie żeby wymusić przebudowanie.
