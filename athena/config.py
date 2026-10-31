"""Chargement et validation de la configuration."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "config.yaml"


@dataclass
class League:
    code: str
    name: str
    country: str
    div: str
    espn: str | None = None
    pinnacle: int | None = None
    sharp: bool = False

    @property
    def label(self) -> str:
        return f"{self.country} - {self.name}"


@dataclass
class Config:
    raw: dict[str, Any]
    leagues: list[League]
    root: Path = ROOT

    # ---- raccourcis typés -------------------------------------------------
    @property
    def history(self) -> dict[str, Any]:
        return self.raw["history"]

    @property
    def model(self) -> dict[str, Any]:
        return self.raw["model"]

    @property
    def betting(self) -> dict[str, Any]:
        return self.raw["betting"]

    @property
    def publish(self) -> dict[str, Any]:
        return self.raw["publish"]

    @property
    def llm(self) -> dict[str, Any]:
        return self.raw.get("llm", {})

    @property
    def cache_dir(self) -> Path:
        p = self.root / self.history.get("cache_dir", "data/cache")
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def db_path(self) -> Path:
        p = self.root / "data" / "athena.db"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def site_dir(self) -> Path:
        p = self.root / self.publish.get("site_dir", "site")
        p.mkdir(parents=True, exist_ok=True)
        return p

    def league(self, code: str) -> League:
        for lg in self.leagues:
            if lg.code == code:
                return lg
        raise KeyError(f"ligue inconnue: {code}")

    def league_by_pinnacle(self, pid: int) -> League | None:
        for lg in self.leagues:
            if lg.pinnacle == pid:
                return lg
        return None

    def league_by_espn(self, slug: str) -> League | None:
        for lg in self.leagues:
            if lg.espn == slug:
                return lg
        return None


def load_config(path: str | Path | None = None) -> Config:
    """Charge config.yaml (+ surcharge par variables d'environnement)."""
    p = Path(path) if path else DEFAULT_CONFIG
    if not p.is_absolute():
        p = ROOT / p
    with open(p, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    leagues = [League(**lg) for lg in raw.get("leagues", [])]

    # surcharges utiles en CI
    if os.getenv("ATHENA_BANKROLL"):
        raw["betting"]["initial_bankroll"] = float(os.environ["ATHENA_BANKROLL"])
    if os.getenv("ATHENA_MIN_EDGE"):
        raw["betting"]["min_edge"] = float(os.environ["ATHENA_MIN_EDGE"])
    if os.getenv("ATHENA_SEASONS_FROM"):
        raw["history"]["seasons_from"] = int(os.environ["ATHENA_SEASONS_FROM"])
    if os.getenv("ATHENA_DISABLE_LLM", "").lower() in {"1", "true", "yes"}:
        raw.setdefault("llm", {})["enabled"] = False

    return Config(raw=raw, leagues=leagues, root=ROOT)


def season_code(start_year: int) -> str:
    """2016 -> '1617' (format football-data.co.uk)."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def season_label(start_year: int) -> str:
    return f"{start_year}/{start_year + 1}"
