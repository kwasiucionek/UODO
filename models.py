"""
Modele danych (Pydantic) i szablony Jinja2 kontekstu LLM.

Wzorce z kursu Software 3.0:
  - QueryDecomposition → Reasoning Step (moduł 4.1)
  - AgentMemory        → Memory Engineering (moduł 2.1)
  - Szablony Jinja2    → Context Engineering (moduł 1.2)
"""

import re
from enum import Enum

from jinja2 import Environment
from pydantic import BaseModel, Field


# ─────────────────────────── MODELE PYDANTIC ─────────────────────

class QueryType(str, Enum):
    DECISION_LOOKUP  = "szukam_decyzji"    # zapytanie o konkretną decyzję/karę
    LEGAL_ARTICLE    = "szukam_przepisu"   # pytanie o artykuł ustawy/RODO
    GENERAL_ANALYSIS = "analiza_ogólna"    # szeroka analiza tematu
    FACTUAL          = "pytanie_faktyczne" # kto/kiedy/ile


class QueryDecomposition(BaseModel):
    """Reasoning Step — LLM dekompozycja pytania PRZED wyszukiwaniem."""

    original_query:        str
    query_type:            QueryType = QueryType.GENERAL_ANALYSIS
    search_keywords:       list[str] = Field(default_factory=list,
        description="Synonimy i pojęcia prawne do wyszukiwania (max 5)")
    gdpr_articles_hint:    list[str] = Field(default_factory=list,
        description="Artykuły RODO które mogą być istotne (np. ['Art. 5', 'Art. 83'])")
    uodo_act_articles_hint: list[str] = Field(default_factory=list,
        description="Artykuły u.o.d.o. które mogą być istotne")
    year_from_hint: int | None = None
    year_to_hint:   int | None = None
    enriched_query: str = Field(description="Rozszerzone zapytanie do wyszukiwania semantycznego")
    reasoning:      str = Field(description="Krótkie uzasadnienie dekompozycji (widoczne w UI)")


class MemoryEntry(BaseModel):
    """Wpis w pamięci epizodycznej."""

    query:                str
    enriched_query:       str
    decomposition_summary: str
    top_signatures: list[str] = []  # sygnatury top decyzji
    top_articles:   list[str] = []  # numery artykułów
    answer_snippet: str = ""        # pierwsze 300 znaków odpowiedzi AI


class AgentMemory(BaseModel):
    """Pamięć epizodyczna sesji — wzorzec z lekcji 2.1 Memory Engineering."""

    entries:     list[MemoryEntry] = []
    max_entries: int = 5

    def add(self, entry: MemoryEntry) -> None:
        self.entries.insert(0, entry)
        self.entries = self.entries[: self.max_entries]

    def find_related(self, query: str) -> list[MemoryEntry]:
        """Prosta heurystyka: wpisy z co najmniej jednym wspólnym słowem kluczowym."""
        q_words = {w.lower() for w in re.split(r"\W+", query) if len(w) > 3}
        return [
            e for e in self.entries
            if q_words & {w.lower() for w in re.split(r"\W+", e.query) if len(w) > 3}
        ]


# ─────────────────────────── SZABLONY JINJA2 ─────────────────────
# Każdy typ dokumentu ma własny szablon — poprawia "zakotwiczenie uwagi" modelu.

_JINJA_ENV = Environment(keep_trailing_newline=True)

# Nagłówek jawnie wymienia WSZYSTKIE typy dokumentów — duże modele (kimi2.5, gpt-4o)
# czytają go dosłownie i bez tej informacji traktują artykuły RODO jako jedyną treść.
TPL_HEADER = _JINJA_ENV.from_string(
    "Poniżej znajdują się dokumenty powiązane z pytaniem: «{{ query }}»\n"
    "Zbiór zawiera trzy typy dokumentów:\n"
    "  1. DECYZJE UODO — decyzje administracyjne Prezesa Urzędu Ochrony Danych Osobowych\n"
    "  2. ARTYKUŁY u.o.d.o. — przepisy ustawy o ochronie danych osobowych (Dz.U. 2019 poz. 1781)\n"
    "  3. ARTYKUŁY RODO — przepisy rozporządzenia (UE) 2016/679 (Dz.Urz. UE L 119/1)\n"
    "Każdy dokument jest wyraźnie oznaczony typem w nagłówku bloku.\n"
    "{% if filter_note %}{{ filter_note }}{% endif %}"
    "{% if memory_note %}{{ memory_note }}{% endif %}"
    "Odpowiadaj na podstawie poniższych dokumentów, ze szczególnym uwzględnieniem DECYZJI UODO.\n"
    "Podawaj sygnatury decyzji [np. DKN.XXX.X.XXXX, ZSOŚS, i in.] i numery artykułów [np. Art. X u.o.d.o.].\n"
)

TPL_DECISION = _JINJA_ENV.from_string(
    "[{{ rank }}] DECYZJA UODO {{ sig }} ({{ date }}, {{ status }})"
    "{% if graph_rel %} [powiązana: {{ graph_rel }}]{% endif %}\n\n"
    "  SYGNATURA:    {{ sig }}\n"
    "  DATA:         {{ date }}\n"
    "  STATUS:       {{ status }}\n"
    "{% if keywords %}  TAGI:         {{ keywords }}\n{% endif %}"
    "{% if acts %}  POWOŁANE AKTY: {{ acts }}\n{% endif %}"
    "  TREŚĆ:\n{{ fragment }}\n"
)

TPL_ACT_ARTICLE = _JINJA_ENV.from_string(
    "[{{ rank }}] USTAWA o ochronie danych osobowych — Art. {{ art_num }}"
    "{% if label_suffix %} {{ label_suffix }}{% endif %}\n\n"
    "  ŹRÓDŁO: Dz.U. 2019 poz. 1781 (u.o.d.o.)\n"
    "  TREŚĆ:\n{{ text }}\n"
)

TPL_GDPR = _JINJA_ENV.from_string(
    "[{{ rank }}] RODO (rozporządzenie 2016/679) — {{ prefix }}\n\n"
    "  ŹRÓDŁO: Dz.Urz. UE L 119/1\n"
    "  TREŚĆ:\n{{ text }}\n"
)

# Kolejność typów w kontekście LLM: decyzje UODO pierwsze, RODO ostatnie
CONTEXT_TYPE_ORDER: dict[str, int] = {
    "uodo_decision":    0,
    "legal_act_article": 1,
    "gdpr_article":     2,
    "gdpr_recital":     3,
}
