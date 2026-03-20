# Qwen3-Embedding i Qwen3-Reranker

https://huggingface.co/Qwen/Qwen3-Reranker-8B

https://huggingface.co/Qwen/Qwen3-Embedding-8B

# Kiedy warto użyć Qwen3-Embedding i Qwen3-Reranker (RTX 5090)

## Qwen3-Embedding — zmiana modelu embeddingowego

Obecny model `sdadas/mmlw-retrieval-roberta-large` jest dobry dla polskiego,
ale **Qwen3-Embedding** warto rozważyć gdy:

- **Zapytania są długie i opisowe** — Qwen3 ma okno 32–40k tokenów vs ~512 tokenów mmlw,
  więc lepiej rozumie kontekst całego pytania
- **Zależy Ci na szybkości indeksowania** — generowanie embeddingów na GPU jest
  kilkanaście razy szybsze niż na CPU, co ma znaczenie przy przebudowie kolekcji
  po dodaniu nowych decyzji
- **Chcesz instruction-aware embeddings** — możesz podać instrukcję dopasowaną do domeny, np.:

  ```
  "Znajdź polskie decyzje administracyjne dotyczące ochrony danych osobowych"
  ```

  co ukierunkowuje wektor semantyczny na kontekst prawny

> ⚠️ **Uwaga:** zmiana modelu embeddingowego wymaga **obowiązkowej przebudowy kolekcji Qdrant**
> (`uodo_indexer.py --rebuild`), ponieważ wymiary wektorów są inne (4096 vs ~1024).

---

## Qwen3-Reranker — dodatkowy etap po wyszukiwaniu

Reranker ocenia każdy wynik wyszukiwania ponownie w kontekście pełnego zapytania
i sortuje je trafniej. Ma sens przede wszystkim gdy:

- **Wyszukiwanie tagowe nie zadziałało** (kroki 1a–1b w `hybrid_search`) i uruchomił się
  semantic search jako fallback — dla zapytań opisowych bez bezpośredniego dopasowania
  frazy do tagu w bazie. Semantic search zwraca wtedy 20 kandydatów,
  a reranker wyłania z nich najlepsze.
- **Zapytania są niejednoznaczne lub wielowątkowe** — reranker rozumie pełny kontekst
  pytania i potrafi odróżnić decyzję naprawdę dotyczącą tematu od takiej,
  która tylko zawiera szukane słowa.

Dla zapytań które trafiają bezpośrednio w tag (np. `"dane genetyczne"` → 26 decyzji)
reranker **nic nie wnosi** — wyniki są już precyzyjne dzięki indeksowaniu tagów.

---

## Zajętość VRAM na RTX 5090 (32 GB)

| Komponent | Model | VRAM |
|---|---|---|
| Embedding | `qwen3-embedding:4b` | ~2.5 GB |
| Reranker | `Qwen3-Reranker-4B` (Q4) | ~4–5 GB |
| Reranker | `Qwen3-Reranker-8B` (BF16) | ~16 GB |
| LLM lokalny | np. `qwen3:14b` | ~8–9 GB |

Wariant zbalansowany: `qwen3-embedding:4b` + `Qwen3-Reranker-4B` + `qwen3:14b` ≈ **15–16 GB**,
zostawiając ponad 16 GB na inne procesy.

---

## Rekomendowana kolejność migracji

1. Zmień embedding na `qwen3-embedding:4b` → przebuduj kolekcję Qdrant
2. Dodaj `Qwen3-Reranker-0.6B` → przetestuj jakość wyników
3. Jeśli jakość dobra → upgrade do `Qwen3-Reranker-4B` lub `8B`
