"""Construction unifiée des offres de marché (historique ET live).

Une ``Offer`` est tout ce dont le moteur de décision a besoin :

    market, selection, line
    price_bet    cote réellement jouable (meilleure dispo, ou moyenne marché)
    price_sharp  cote de clôture sharp (référence CLV)
    p_sharp      probabilité dé-viggée de la référence sharp
    book         nom du bookmaker (live multi-books)
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..logging_conf import get_logger
from ..models.markets import ou_line_from_market
from .devig import devig_named

log = get_logger("betting.offers")

PRICE_PREFIXES = {
    "avg": ("avg", "avgc", "b365"),
    "max": ("max", "maxc", "avg", "b365"),
    "avgc": ("avgc", "avg", "b365"),
}
SHARP_PREFIXES = ("psc", "avgc", "avg", "b365")


@dataclass
class Offer:
    market: str
    selection: str
    line: float | None
    price_bet: float
    price_sharp: float | None
    p_sharp: float | None
    book: str = "market"

    def clv(self) -> float | None:
        if self.price_sharp and self.price_sharp > 1:
            return round(self.price_bet / self.price_sharp - 1.0, 5)
        return None


def _num(v) -> float | None:
    try:
        f = float(v)
        return f if f > 1.01 else None
    except (TypeError, ValueError):
        return None


# ==========================================================================
# 1) Données historiques (football-data.co.uk)
# ==========================================================================
def _row_odds(row, prefix: str) -> dict[str, float] | None:
    vals = {}
    for k in ("home", "draw", "away"):
        v = _num(row.get(f"odds_{prefix}_{k}"))
        if v is None:
            return None
        vals[k] = v
    return vals


def _sharp_1x2(row) -> tuple[dict[str, float] | None, dict[str, float]]:
    """(prix sharp, probabilité sharp) pour 1X2."""
    for prefix in SHARP_PREFIXES:
        odds = _row_odds(row, prefix)
        if odds:
            return odds, devig_named([odds["home"], odds["draw"], odds["away"]],
                                     ["home", "draw", "away"], "power")
    return None, {}


def _ou_odds(row, prefix: str) -> dict[str, float] | None:
    over = _num(row.get(f"odds_ou_{prefix}_over"))
    under = _num(row.get(f"odds_ou_{prefix}_under"))
    return {"over": over, "under": under} if over and under else None


def offers_from_history(row: pd.Series, markets: list[str],
                        price_mode: str = "avg") -> list[Offer]:
    """Offres jouables reconstituées depuis une ligne football-data."""
    out: list[Offer] = []
    bet_1x2, sharp_odds, p_sharp = None, None, {}
    pref = PRICE_PREFIXES.get(price_mode, PRICE_PREFIXES["avg"])
    for prefix in pref:
        bet_1x2 = _row_odds(row, prefix)
        if bet_1x2:
            break
    sharp_odds, p_sharp = _sharp_1x2(row)

    if "1X2" in markets and bet_1x2 and p_sharp:
        for k in ("home", "draw", "away"):
            out.append(Offer("1X2", k, None, bet_1x2[k],
                             (sharp_odds or {}).get(k), p_sharp[k], "market"))
    if "DC" in markets and bet_1x2 and p_sharp:
        pairs = {"1X": ("home", "draw"), "12": ("home", "away"), "X2": ("draw", "away")}
        for sel, (a, b) in pairs.items():
            price = 1.0 / (1.0 / bet_1x2[a] + 1.0 / bet_1x2[b])
            out.append(Offer("DC", sel, None, round(price, 3), None,
                             p_sharp[a] + p_sharp[b], "market"))
    if any(m.upper().startswith("OU") for m in markets):
        bet = _ou_odds(row, "max" if price_mode == "max" else "avg") or _ou_odds(row, "avgc")
        sharp = _ou_odds(row, "psc") or _ou_odds(row, "avgc")
        if bet and sharp:
            dev = devig_named([sharp["over"], sharp["under"]], ["over", "under"], "power")
            for k in ("over", "under"):
                out.append(Offer("OU25", k, 2.5, bet[k], sharp[k], dev[k], "market"))
    if "AH" in markets:
        line = row.get("odds_ah_line")
        p_bet = {k: _num(row.get(f"odds_ah_{'maxc' if price_mode == 'max' else 'avg'}_{k}"))
                 for k in ("home", "away")}
        p_sharp = {k: _num(row.get(f"odds_ah_psc_{k}")) for k in ("home", "away")}
        if not all(p_sharp.values()):
            p_sharp = {k: _num(row.get(f"odds_ah_maxc_{k}")) for k in ("home", "away")}
        try:
            line = float(line)
        except (TypeError, ValueError):
            line = None
        if line is not None and all(p_bet.values()) and all(p_sharp.values()):
            dev = devig_named([p_sharp["home"], p_sharp["away"]], ["home", "away"], "power")
            for k in ("home", "away"):
                out.append(Offer("AH", k, line, p_bet[k], p_sharp[k], dev[k], "market"))
    return [o for o in out if o.price_bet and o.price_bet > 1.01]


# ==========================================================================
# 2) Live (board Pinnacle + multi-books)
# ==========================================================================
def offers_from_live(row: dict, markets: list[str], min_books: int = 3) -> list[Offer]:
    """Offres d'un match à venir.

    - référence de probabilité : Pinnacle dé-viggé (sharp)
    - prix jouable : meilleure cote multi-books si disponible, sinon Pinnacle
    """
    out: list[Offer] = []
    pinn = row.get("pinnacle_odds") or {}
    best = row.get("best_odds") or {}
    books = row.get("books") or {}
    multi = len(best) == 3 and row.get("n_books", 0) >= min_books

    p_sharp_1x2: dict[str, float] = {}
    if len(pinn) == 3:
        p_sharp_1x2 = devig_named([pinn["home"], pinn["draw"], pinn["away"]],
                                  ["home", "draw", "away"], "power")
        prices = best if multi else pinn
        for k in ("home", "draw", "away"):
            price = _num(prices.get(k))
            if price:
                out.append(Offer("1X2", k, None, price, pinn[k], p_sharp_1x2[k],
                                 books.get(k, "pinnacle") if multi else "pinnacle"))

    if "DC" in markets and p_sharp_1x2:
        prices = best if multi else pinn
        pairs = {"1X": ("home", "draw"), "12": ("home", "away"), "X2": ("draw", "away")}
        for sel, (a, b) in pairs.items():
            pa, pb = _num(prices.get(a)), _num(prices.get(b))
            if not pa or not pb:
                continue
            out.append(Offer("DC", sel, None, round(1 / (1 / pa + 1 / pb), 3), None,
                             p_sharp_1x2[a] + p_sharp_1x2[b], "pinnacle"))

    # ---- totals (over/under) -------------------------------------------
    if any(m.upper().startswith("OU") for m in markets):
        ptot = row.get("pinnacle_totals") or {}     # {"2.5": {"over":..,"under":..}}
        btot = row.get("best_totals") or {}         # {"2.5": {"over":..,"under":..}}
        wanted = [ou_line_from_market(m) for m in markets if m.upper().startswith("OU")]
        wanted = [w for w in wanted if w] or [2.5]
        avail = {}
        for ln_key, prices in ptot.items():
            try:
                avail[float(ln_key)] = prices
            except (TypeError, ValueError):
                continue
        lines = []
        for target in wanted:
            if avail:
                lines.append(min(avail, key=lambda x: (abs(x - target), abs(x - round(x)))))
        for ln in dict.fromkeys(lines):
            sharp_prices = avail[ln]
            ln_key = f"{ln:g}"
            ov, un = _num(sharp_prices.get("over")), _num(sharp_prices.get("under"))
            if not ov or not un:
                continue
            dev = devig_named([ov, un], ["over", "under"], "power")
            bet = (btot.get(ln_key) or {}) if multi else {}
            for k in ("over", "under"):
                price = _num(bet.get(k)) or _num(sharp_prices.get(k))
                if price:
                    name = "OU25" if abs(ln - 2.5) < 0.4 else f"OU{ln:g}"
                    out.append(Offer(name, k, ln, price, sharp_prices.get(k), dev[k],
                                     "market" if bet else "pinnacle"))

    # ---- handicap asiatique (ligne principale, quart inclus) ------------
    if "AH" in markets:
        ph = row.get("pinnacle_handicaps") or {}    # {"-0.75": {"home":..,"away":..}}
        bh = row.get("best_handicaps") or {}
        lines = {}
        for ln_key, prices in ph.items():
            try:
                ln = float(ln_key)
            except (TypeError, ValueError):
                continue
            h, a = _num(prices.get("home")), _num(prices.get("away"))
            if h and a:
                lines[ln] = (h, a)
        if lines:
            main = min(lines, key=lambda k: (abs(k), abs(k - round(k))))
            h, a = lines[main]
            dev = devig_named([h, a], ["home", "away"], "power")
            bet = (bh.get(f"{main:g}") or bh.get(main) or {}) if multi else {}
            for k, sharp_price in (("home", h), ("away", a)):
                price = _num(bet.get(k)) or sharp_price
                out.append(Offer("AH", k, main, price, sharp_price, dev[k],
                                 "market" if bet else "pinnacle"))
    return out
