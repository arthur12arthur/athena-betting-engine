"""Historique de résultats + cotes de fermeture (football-data.co.uk).

Source 100 % gratuite, sans clé, sans limite : ~20 saisons de résultats
et de cotes de fermeture (moyenne marché, Pinnacle, Bet365...).
C'est la colonne vertébrale de l'entraînement et du backtest.
"""
from __future__ import annotations

import io
import re
from datetime import datetime

import pandas as pd

from ..config import Config, season_code
from ..http import HttpClient
from ..logging_conf import get_logger

log = get_logger("source.fdcsv")

BASE = "https://www.football-data.co.uk/mmz4281/{season}/{div}.csv"

# ---------------------------------------------------------------
# Cotes : on distingue les cotes "pré-clôture" (celles qu'on peut
# réellement jouer) des cotes de clôture (référence sharp / CLV).
# ---------------------------------------------------------------
ODDS_MAP = {
    "avg":  ("AvgH", "AvgD", "AvgA"),        # moyenne marché pré-clôture
    "max":  ("MaxH", "MaxD", "MaxA"),        # meilleure cote pré-clôture
    "psc":  ("PSCH", "PSCD", "PSCA"),        # Pinnacle clôture (sharp)
    "avgc": ("AvgCH", "AvgCD", "AvgCA"),     # moyenne clôture
    "maxc": ("MaxCH", "MaxCD", "MaxCA"),     # meilleure clôture
    "b365": ("B365H", "B365D", "B365A"),
}

# Over/Under 2,5 et handicap asiatique (clôture + pré-clôture)
EXTRA_ODDS = {
    "ou_avgc_over": "AvgC>2.5", "ou_avgc_under": "AvgC<2.5",
    "ou_avg_over": "Avg>2.5", "ou_avg_under": "Avg<2.5",
    "ou_psc_over": "PC>2.5", "ou_psc_under": "PC<2.5",
    "ou_maxc_over": "MaxC>2.5", "ou_maxc_under": "MaxC<2.5",
    "ah_line": "AHCh",
    "ah_avg_home": "AvgCAHH", "ah_avg_away": "AvgCAHA",
    "ah_psc_home": "PCAHH", "ah_psc_away": "PCAHA",
    "ah_maxc_home": "MaxCAHH", "ah_maxc_away": "MaxCAHA",
}

RENAME = {
    "HomeTeam": "home",
    "AwayTeam": "away",
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "FTR": "result",
    "HTHG": "ht_home_goals",
    "HTAG": "ht_away_goals",
    "HxG": "xg_home",
    "AxG": "xg_away",
    "HS": "shots_home",
    "AS": "shots_away",
    "HST": "sot_home",
    "AST": "sot_away",
    "HC": "corners_home",
    "AC": "corners_away",
    "HY": "yellow_home",
    "AY": "yellow_away",
    "HR": "red_home",
    "AR": "red_away",
}

NUMERIC = [
    "home_goals", "away_goals", "ht_home_goals", "ht_away_goals", "xg_home", "xg_away",
    "shots_home", "shots_away", "sot_home", "sot_away", "corners_home", "corners_away",
    "yellow_home", "yellow_away", "red_home", "red_away",
]


def _parse_dates(series: pd.Series) -> pd.Series:
    """football-data mélange jj/mm/aa et jj/mm/aaaa."""
    s = series.astype(str).str.strip()

    def one(v: str) -> datetime | None:
        for fmt in ("%d/%m/%Y", "%d/%m/%y"):
            try:
                return datetime.strptime(v, fmt)
            except ValueError:
                continue
        return None

    return pd.to_datetime(s.map(one), errors="coerce")


def fetch_season(div: str, start_year: int, client: HttpClient, cache_hours: float = 12) -> pd.DataFrame:
    """Télécharge (avec cache) une saison d'une division."""
    season = season_code(start_year)
    url = BASE.format(season=season, div=div)
    text = client.get(url, cache_hours=cache_hours, raw=True, ext="csv")
    if not text or "HomeTeam" not in text[:2000]:
        return pd.DataFrame()
    text = text.lstrip("\ufeff")
    df = pd.read_csv(io.StringIO(text), encoding="utf-8", on_bad_lines="skip")
    if df.empty or "HomeTeam" not in df.columns:
        return df
    df["div"] = div
    df["season_start"] = start_year
    return df


def _extract_odds(df: pd.DataFrame) -> pd.DataFrame:
    """Extrait toutes les familles de cotes (1X2, O/U 2,5, handicap)."""
    out = pd.DataFrame(index=df.index)
    for prefix, cols in ODDS_MAP.items():
        for label, col in zip(("home", "draw", "away"), cols, strict=False):
            out[f"odds_{prefix}_{label}"] = pd.to_numeric(df.get(col), errors="coerce")
    for name, col in EXTRA_ODDS.items():
        if col in df.columns:
            out[f"odds_{name}"] = pd.to_numeric(df[col], errors="coerce")
        else:
            out[f"odds_{name}"] = pd.NA
    return out


def normalize(raw: pd.DataFrame) -> pd.DataFrame:
    """DataFrame brut football-data -> schéma canonique ATHENA."""
    if raw.empty:
        return raw
    df = raw.copy()
    # certaines saisons anciennes n'ont pas toutes les colonnes (xG, tirs...) :
    for col in RENAME:
        if col not in df.columns:
            df[col] = pd.NA
    df = df.rename(columns=RENAME)
    df["date"] = _parse_dates(df["Date"]) if "Date" in df.columns else pd.NaT
    df = df.dropna(subset=["date", "home", "away"])
    for col in NUMERIC:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    odds = _extract_odds(df)
    df = pd.concat([df, odds], axis=1)

    # marchés dérivés (issues réelles)
    df["total_goals"] = df["home_goals"] + df["away_goals"]
    df["btts"] = ((df["home_goals"] > 0) & (df["away_goals"] > 0)).astype("float")
    df.loc[df["home_goals"].isna(), "btts"] = None
    df["over25"] = (df["total_goals"] > 2.5).astype("float")
    df.loc[df["home_goals"].isna(), "over25"] = None

    df["home_goals"] = df["home_goals"].astype("Int64")
    df["away_goals"] = df["away_goals"].astype("Int64")
    df["match_id"] = (
        df["div"].astype(str) + "_" + df["date"].dt.strftime("%Y%m%d") + "_"
        + df["home"].map(_slug) + "_" + df["away"].map(_slug)
    )
    keep = [
        "match_id", "div", "season_start", "date", "home", "away", "result",
        "home_goals", "away_goals", "ht_home_goals", "ht_away_goals", "xg_home", "xg_away",
        "shots_home", "shots_away", "sot_home", "sot_away", "corners_home", "corners_away",
        "total_goals", "btts", "over25",
    ] + [c for c in df.columns if c.startswith("odds_")]
    out = df[keep].copy()
    out["league"] = out["div"]
    return out.sort_values("date").reset_index(drop=True)


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(name).lower()).strip("-")


def load_history(cfg: Config, client: HttpClient, seasons_from: int | None = None,
                 seasons_to: int | None = None) -> pd.DataFrame:
    """Charge l'historique complet (toutes ligues configurées)."""
    start = seasons_from or int(cfg.history.get("seasons_from", 2016))
    now = datetime.utcnow()
    # saison en cours : commence en juillet/août
    current = now.year if now.month >= 7 else now.year - 1
    end = seasons_to or current
    cache_hours = float(cfg.history.get("refresh_hours", 12))
    frames = []
    for lg in cfg.leagues:
        for year in range(start, end + 1):
            raw = fetch_season(lg.div, year, client, cache_hours=cache_hours)
            if raw is None or raw.empty:
                continue
            df = normalize(raw)
            if not df.empty:
                frames.append(df)
    if not frames:
        return pd.DataFrame()
    hist = pd.concat(frames, ignore_index=True)
    hist = hist.drop_duplicates(subset=["match_id"]).sort_values("date").reset_index(drop=True)
    log.info("historique chargé : %d matchs, %d ligues, saisons %s-%s",
             len(hist), hist["league"].nunique(), start, end)
    return hist


def load_current_season_fixtures(cfg: Config, client: HttpClient) -> pd.DataFrame:
    """Matchs déjà joués de la saison en cours (pour se mettre à jour vite)."""
    return load_history(cfg, client, seasons_from=None, seasons_to=None)
