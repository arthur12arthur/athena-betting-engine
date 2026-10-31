"""Offres de marché : cotes réelles + probabilités "sharp" de référence.

Ce module traduit une ligne de données (historique football-data ou snapshot
Pinnacle) en offres comparables :

    Offer(market, selection, line, price_bet, price_sharp, p_sharp)

  - ``price_bet``    : cote à laquelle ON peut parier (pré-clôture, moyenne
                       marché ; ou cote Pinnacle en production)
  - ``price_sharp``  : cote de clôture sharp (Pinnacle) -> référence CLV
  - ``p_sharp``      : probabilité de clôture dé-viggée (estimateur du "vrai" p)
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .devig import devig, devig_named

EPS = 1e-9


@dataclass
class Offer:
    market: str
    selection: str
    line: float | None
    price_bet: float
    price_sharp: float | None
    p_sharp: float | None

    def clv(self) -> float | None:
        if self.price_sharp and self.price_sharp > 1:
            return round(self.price_bet / self.price_sharp - 1.0, 5)
        return None


def _num(v) -> float | None:
    try:
        f = float(v)
        return f if f > 1.0 else None
    except (TypeError, ValueError):
        return None


def _sharp_1x2(row: pd.Series) -> dict[str, float] | None:
    """Pinnacle closing sinon moyenne de clôture (de-viggée en 'power')."""
    for prefix in ("psc", "avgc", "avg", "b365"):
        o = [_num(row.get(f"odds_{prefix}_{k}")) for k in ("home", "draw", "away")]
        if all(o):
            return devig([float(x) for x in o], "power")
    return None


PRICE_PREFIXES = {
    "avg": ("avg", "avgc", "b365"),          # moyenne marché pré-clôture
    "max": ("max", "maxc", "avg", "b365"),   # meilleure cote dispo (line shopping)
    "avgc": ("avgc", "avg", "b365"),         # moyenne de clôture
}


def _bet_1x2(row: pd.Series, price_mode: str = "max") -> dict[str, float]:
    for prefix in PRICE_PREFIXES.get(price_mode, PRICE_PREFIXES["max"]):
        o = [_num(row.get(f"odds_{prefix}_{k}")) for k in ("home", "draw", "away")]
        if all(o):
            return {k: float(x) for k, x in zip(("home", "draw", "away"), o, strict=False)}
    return {}


def _ou(row: pd.Series, prefix: str) -> dict[str, float] | None:
    over = _num(row.get(f"odds_ou_{prefix}_over"))
    under = _num(row.get(f"odds_ou_{prefix}_under"))
    if over and under:
        return {"over": over, "under": under}
    return None


def build_offers(row: pd.Series, markets: list[str], price_mode: str = "max") -> list[Offer]:
    """Construit toutes les offres jouables d'un match (données historiques).

    ``price_mode`` : "max" = meilleure cote disponible (line shopping),
    "avg" = cote moyenne du marché (plus conservateur).
    """
    offers: list[Offer] = []
    p_sharp_1x2 = _sharp_1x2(row)
    bet_1x2 = _bet_1x2(row, price_mode)

    if "1X2" in markets and bet_1x2 and p_sharp_1x2:
        sharp_odds = bet_1x2
        for prefix in ("psc", "avgc"):
            o = [_num(row.get(f"odds_{prefix}_{k}")) for k in ("home", "draw", "away")]
            if all(o):
                sharp_odds = {k: float(x) for k, x in zip(("home", "draw", "away"), o, strict=False)}
                break
        for k in ("home", "draw", "away"):
            offers.append(Offer("1X2", k, None, bet_1x2[k], sharp_odds[k], p_sharp_1x2[k]))

    if "DC" in markets and bet_1x2 and p_sharp_1x2:
        pairs = {"1X": ("home", "draw"), "12": ("home", "away"), "X2": ("draw", "away")}
        for sel, (a, b) in pairs.items():
            price = 1.0 / (1.0 / bet_1x2[a] + 1.0 / bet_1x2[b])
            p = p_sharp_1x2[a] + p_sharp_1x2[b]
            offers.append(Offer("DC", sel, None, round(price, 3), None, p))

    if "OU25" in markets:
        p_bet = (_ou(row, "maxc") or _ou(row, "avgc")) if price_mode == "max" else (_ou(row, "avg") or _ou(row, "avgc"))
        p_sh = _ou(row, "psc") or _ou(row, "avgc")
        if p_bet and p_sh:
            dev = devig_named([p_sh["over"], p_sh["under"]], ["over", "under"], "power")
            for k in ("over", "under"):
                offers.append(Offer("OU25", k, 2.5, p_bet[k], p_sh[k], dev[k]))

    if "AH" in markets:
        line = row.get("odds_ah_line")
        try:
            line = float(line)
        except (TypeError, ValueError):
            line = None
        if line is not None and abs(line * 2 - round(line * 2)) < 1e-9:
            p_bet = {k: _num(row.get(f"odds_ah_{'maxc' if price_mode == 'max' else 'avg'}_{k}")) for k in ("home", "away")}
            p_sh = {k: _num(row.get(f"odds_ah_psc_{k}")) for k in ("home", "away")}
            if not all(p_sh.values()):
                p_sh = {k: _num(row.get(f"odds_ah_maxc_{k}")) for k in ("home", "away")}
            if all(p_bet.values()) and all(p_sh.values()):
                dev = devig_named([float(p_sh["home"]), float(p_sh["away"])], ["home", "away"], "power")
                offers.append(Offer("AH", "home", line, float(p_bet["home"]), float(p_sh["home"]), dev["home"]))
                offers.append(Offer("AH", "away", line, float(p_bet["away"]), float(p_sh["away"]), dev["away"]))
    return [o for o in offers if o.price_bet and o.price_bet > 1.01]
