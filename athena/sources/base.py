"""Types communs aux sources de données."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Fixture:
    """Un match (à venir ou passé) au format canonique ATHENA."""

    league: str                 # code ATHENA (E0, SP1, ...)
    date_utc: str               # ISO 8601
    home: str                   # nom brut source
    away: str
    source: str
    external_id: str | None = None
    status: str = "scheduled"   # scheduled | live | finished | postponed
    home_goals: int | None = None
    away_goals: int | None = None
    extra: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.league}|{self.home}|{self.away}"


@dataclass
class OddsSnapshot:
    """Cotes d'un bookmaker pour un match (format décimale)."""

    book: str
    source: str
    ts: str
    one_x_two: dict[str, float] = field(default_factory=dict)          # home/draw/away
    totals: dict[float, dict[str, float]] = field(default_factory=dict)  # 2.5 -> over/under
    handicaps: dict[float, dict[str, float]] = field(default_factory=dict)  # -0.5 -> home/away

    @property
    def overround(self) -> float:
        if len(self.one_x_two) < 3:
            return 0.0
        return sum(1.0 / v for v in self.one_x_two.values() if v and v > 0)

    def is_usable(self) -> bool:
        return len(self.one_x_two) == 3 and self.overround > 1.0
