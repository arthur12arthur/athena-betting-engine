"""Construction des features (aucune fuite du futur).

Un seul passage chronologique : pour chaque match, on calcule les variables
UNIQUEMENT à partir de ce qui s'est passé avant le coup d'envoi, puis on met
à jour l'état (Elo, forme, xG, repos, congestion). Ce module est utilisé à
l'identique pour l'entraînement, le backtest et la production — c'est ce qui
rend le backtest honnête.

Familles de variables :
  - Elo (pondéré par l'écart de buts, régression inter-saisons)
  - forme (points/buts/xG sur 5 et 10 matchs)
  - spécificité du terrain (domicile / extérieur)
  - repos et congestion (jours depuis le dernier match, matchs en 14 jours)
  - contexte (journée, avancement de saison, montée)
"""
from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime

import numpy as np
import pandas as pd

from ..logging_conf import get_logger

log = get_logger("etl.features")

FEATURES: list[str] = [
    "elo_home", "elo_away", "elo_diff",
    "form5_home", "form5_away", "form10_home", "form10_away",
    "gf5_home", "ga5_home", "gf5_away", "ga5_away",
    "gf10_home", "ga10_home", "gf10_away", "ga10_away",
    "venue_gf_home", "venue_ga_home", "venue_gf_away", "venue_ga_away",
    "xg5_for_home", "xg5_ag_home", "xg5_for_away", "xg5_ag_away",
    "sot5_for_home", "sot5_ag_home", "sot5_for_away", "sot5_ag_away",
    "rest_home", "rest_away", "load14_home", "load14_away",
    "matchday", "season_progress", "is_promoted_home", "is_promoted_away",
    "league_id",
]

CATEGORICAL = ["league_id"]


class EloRating:
    """Elo maison : K adaptatif, multiplicateur d'écart de buts, régression
    vers la moyenne entre saisons (les effectifs changent)."""

    def __init__(self, base: float = 1500.0, k: float = 20.0, hfa: float = 62.0,
                 season_regression: float = 0.72):
        self.base = base
        self.k = k
        self.hfa = hfa
        self.season_regression = season_regression
        self.ratings: dict[str, float] = {}
        self.last_season: dict[str, int] = {}

    def get(self, team: str) -> float:
        return self.ratings.get(team, self.base)

    def expected(self, home: str, away: str) -> float:
        diff = self.get(home) + self.hfa - self.get(away)
        return 1.0 / (1.0 + 10 ** (-diff / 400.0))

    def start_season(self, team: str, season: int) -> None:
        prev = self.last_season.get(team)
        if prev is not None and season != prev:
            self.ratings[team] = self.base + self.season_regression * (self.get(team) - self.base)
        self.last_season[team] = season

    def update(self, home: str, away: str, home_goals: int, away_goals: int) -> None:
        exp_home = self.expected(home, away)
        if home_goals > away_goals:
            score = 1.0
        elif home_goals == away_goals:
            score = 0.5
        else:
            score = 0.0
        gd = abs(home_goals - away_goals)
        mult = 1.0 if gd <= 1 else (1.5 if gd == 2 else (11.0 + gd) / 8.0)
        delta = self.k * mult * (score - exp_home)
        self.ratings[home] = self.get(home) + delta
        self.ratings[away] = self.get(away) - delta


def _mean(values: list[float], default: float = 0.0) -> float:
    return float(np.mean(values)) if values else default


class FeatureBuilder:
    MAXLEN = 12

    def __init__(self, elo: EloRating | None = None, league_map: dict[str, int] | None = None):
        self.elo = elo or EloRating()
        self.league_map = league_map or {}
        self.reset()

    def reset(self) -> None:
        self.history: dict[str, deque] = defaultdict(lambda: deque(maxlen=self.MAXLEN))
        self.last_date: dict[str, datetime | None] = defaultdict(lambda: None)
        self.recent_dates: dict[str, deque] = defaultdict(lambda: deque(maxlen=8))
        self.seasons_seen: dict[str, set] = defaultdict(set)

    # ------------------------------------------------------------------ util
    def _team_features(self, team: str, venue: str, as_of: pd.Timestamp) -> dict[str, float]:
        h = list(self.history.get(team, []))
        out: dict[str, float] = {}
        last5, last10 = h[-5:], h[-10:]
        out["form5"] = _mean([x["pts"] for x in last5], 1.2)
        out["form10"] = _mean([x["pts"] for x in last10], 1.2)
        out["gf5"] = _mean([x["gf"] for x in last5], 1.35)
        out["ga5"] = _mean([x["ga"] for x in last5], 1.35)
        out["gf10"] = _mean([x["gf"] for x in last10], 1.35)
        out["ga10"] = _mean([x["ga"] for x in last10], 1.35)
        same_venue = [x for x in h if x["venue"] == venue][-5:]
        out["venue_gf"] = _mean([x["gf"] for x in same_venue], 1.35)
        out["venue_ga"] = _mean([x["ga"] for x in same_venue], 1.35)
        xg_for = [x["xg_for"] for x in last5 if x["xg_for"] is not None]
        xg_ag = [x["xg_ag"] for x in last5 if x["xg_ag"] is not None]
        out["xg5_for"] = _mean(xg_for, out["gf5"])
        out["xg5_ag"] = _mean(xg_ag, out["ga5"])
        sot_for = [x["sot"] for x in last5 if x["sot"] is not None]
        sot_ag = [x["sot_ag"] for x in last5 if x["sot_ag"] is not None]
        out["sot5_for"] = _mean(sot_for, 4.5)
        out["sot5_ag"] = _mean(sot_ag, 4.5)
        ld = self.last_date.get(team)
        out["rest"] = 14.0 if ld is None else float(np.clip((as_of - pd.Timestamp(ld)).days, 2, 30))
        out["load14"] = float(sum(1 for d in self.recent_dates.get(team, [])
                                  if (as_of - pd.Timestamp(d)).days <= 14))
        return out

    def _row(self, row, matchday: int, season_progress: float) -> dict:
        as_of = pd.Timestamp(row["date"])
        home, away = row["home"], row["away"]
        fh = self._team_features(home, "home", as_of)
        fa = self._team_features(away, "away", as_of)
        elo_h, elo_a = self.elo.get(home), self.elo.get(away)
        out = {
            "elo_home": elo_h, "elo_away": elo_a,
            "elo_diff": elo_h + self.elo.hfa - elo_a,
            "form5_home": fh["form5"], "form5_away": fa["form5"],
            "form10_home": fh["form10"], "form10_away": fa["form10"],
            "gf5_home": fh["gf5"], "ga5_home": fh["ga5"],
            "gf5_away": fa["gf5"], "ga5_away": fa["ga5"],
            "gf10_home": fh["gf10"], "ga10_home": fh["ga10"],
            "gf10_away": fa["gf10"], "ga10_away": fa["ga10"],
            "venue_gf_home": fh["venue_gf"], "venue_ga_home": fh["venue_ga"],
            "venue_gf_away": fa["venue_gf"], "venue_ga_away": fa["venue_ga"],
            "xg5_for_home": fh["xg5_for"], "xg5_ag_home": fh["xg5_ag"],
            "xg5_for_away": fa["xg5_for"], "xg5_ag_away": fa["xg5_ag"],
            "sot5_for_home": fh["sot5_for"], "sot5_ag_home": fh["sot5_ag"],
            "sot5_for_away": fa["sot5_for"], "sot5_ag_away": fa["sot5_ag"],
            "rest_home": fh["rest"], "rest_away": fa["rest"],
            "load14_home": fh["load14"], "load14_away": fa["load14"],
            "matchday": matchday, "season_progress": season_progress,
            "is_promoted_home": int(len(self.seasons_seen.get(home, set())) <= 1
                                    and len(self.history.get(home, [])) > 0),
            "is_promoted_away": int(len(self.seasons_seen.get(away, set())) <= 1
                                    and len(self.history.get(away, [])) > 0),
            "league_id": self.league_map.get(str(row.get("league", "")), 0),
        }
        return out

    def _update(self, row) -> None:
        home, away = row["home"], row["away"]
        hg, ag = int(row["home_goals"]), int(row["away_goals"])
        d = pd.Timestamp(row["date"])
        season = int(row.get("season_start") or d.year)
        for team in (home, away):
            self.elo.start_season(team, season)
            self.seasons_seen[team].add(season)
        pts_h = 3 if hg > ag else (1 if hg == ag else 0)
        pts_a = 3 if ag > hg else (1 if hg == ag else 0)
        xg_h, xg_a = row.get("xg_home"), row.get("xg_away")
        sot_h, sot_a = row.get("sot_home"), row.get("sot_away")
        self.history[home].append({"date": d, "gf": hg, "ga": ag, "pts": pts_h, "venue": "home",
                                   "xg_for": None if pd.isna(xg_h) else float(xg_h),
                                   "xg_ag": None if pd.isna(xg_a) else float(xg_a),
                                   "sot": None if pd.isna(sot_h) else float(sot_h),
                                   "sot_ag": None if pd.isna(sot_a) else float(sot_a)})
        self.history[away].append({"date": d, "gf": ag, "ga": hg, "pts": pts_a, "venue": "away",
                                   "xg_for": None if pd.isna(xg_a) else float(xg_a),
                                   "xg_ag": None if pd.isna(xg_h) else float(xg_h),
                                   "sot": None if pd.isna(sot_a) else float(sot_a),
                                   "sot_ag": None if pd.isna(sot_h) else float(sot_h)})
        self.last_date[home] = d
        self.last_date[away] = d
        self.recent_dates[home].append(d)
        self.recent_dates[away].append(d)
        self.elo.update(home, away, hg, ag)

    # ------------------------------------------------------------- interface
    def build(self, matches: pd.DataFrame, targets: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Retourne (train_features, target_features).

        ``matches`` : matchs terminés, triés par date (l'état est reconstruit).
        ``targets`` : matchs à venir (colonnes league/date/home/away).
        """
        m = matches.dropna(subset=["home_goals", "away_goals"]).sort_values("date")
        rows: list[dict] = []
        matchday_counter: dict[str, int] = defaultdict(int)

        for r in m.itertuples(index=False):
            row = r._asdict()
            league = str(row.get("league"))
            season = row.get("season_start")
            key = f"{league}-{season}"
            matchday_counter[key] += 1
            md = matchday_counter[key]
            feats = self._row(row, md, min(md / 38.0, 1.0))
            feats.update({
                "match_id": row.get("match_id"), "date": row.get("date"), "league": league,
                "season_start": season, "home": row.get("home"), "away": row.get("away"),
                "result": row.get("result"),
                "result_code": {"H": 0, "D": 1, "A": 2}.get(str(row.get("result")), np.nan),
                "home_goals": float(row["home_goals"]), "away_goals": float(row["away_goals"]),
                **{k: v for k, v in row.items() if str(k).startswith("odds_")},
                "over25": row.get("over25"), "btts": row.get("btts"),
            })
            rows.append(feats)
            self._update(row)

        train = pd.DataFrame(rows)

        tgt_rows: list[dict] = []
        if targets is not None and len(targets) > 0:
            t = targets.copy()
            t["date"] = pd.to_datetime(t["date"])
            for r in t.sort_values("date").itertuples(index=False):
                row = r._asdict()
                feats = self._row(row, 0, 0.5)
                feats.update({
                    "match_id": row.get("match_id"), "date": row.get("date"),
                    "league": str(row.get("league")), "home": row.get("home"), "away": row.get("away"),
                })
                tgt_rows.append(feats)
        return train, pd.DataFrame(tgt_rows)


def prepare(history: pd.DataFrame, fixtures: pd.DataFrame | None = None,
            league_map: dict[str, int] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Point d'entrée : construit features train + fixtures."""
    fb = FeatureBuilder(league_map=league_map or {})
    train, tgt = fb.build(history, fixtures)
    log.info("features : %d matchs, %d variables, %d fixtures", len(train), len(FEATURES), len(tgt))
    return train, tgt
