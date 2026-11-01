"""Règlement des paris, tous marchés, y compris lignes de handicap en quart.

Renvoie directement le P&L, ce qui évite les erreurs de conversion entre
« gagné/perdu » et « gagné à moitié / perdu à moitié » :

  - ligne entière  : push possible (mise rendue)
  - demi-ligne     : jamais de push
  - ligne quart    : demi-gain ou demi-perte si l'écart tombe pile
                     (ex. handicap -0,75 avec un but d'avance)
"""
from __future__ import annotations

from ..models.markets import is_quarter_line, ou_line_from_market

WON, LOST, PUSH, HALF_WON, HALF_LOST, VOID = (
    "won", "lost", "push", "half_won", "half_lost", "void")

SETTLED_STATUSES = {WON, LOST, PUSH, HALF_WON, HALF_LOST}
WIN_STATUSES = {WON, HALF_WON}


def _one_x_two(hg: int, ag: int) -> str:
    return "home" if hg > ag else ("draw" if hg == ag else "away")


def settle_pnl(market: str, selection: str, line: float | None, stake: float,
               price: float, hg: int, ag: int) -> tuple[float, str]:
    """P&L net et statut d'un pari. ``price`` = cote décimale jouée."""
    stake = float(stake)
    gain = stake * (float(price) - 1.0)
    market = (market or "").upper()

    # ---- marchés discrets -------------------------------------------------
    if market == "1X2":
        return (gain, WON) if selection == _one_x_two(hg, ag) else (-stake, LOST)
    if market == "DC":
        res = _one_x_two(hg, ag)
        wins = {"1X": res in {"home", "draw"}, "12": res in {"home", "away"},
                "X2": res in {"draw", "away"}}
        return (gain, WON) if wins.get(selection) else (-stake, LOST)
    if market == "BTTS":
        yes = hg > 0 and ag > 0
        return (gain, WON) if (selection == "yes") == yes else (-stake, LOST)

    # ---- marchés à écart ------------------------------------------------
    if market.startswith("OU"):
        ln = float(line) if line is not None else (ou_line_from_market(market) or 2.5)
        total = hg + ag
        d = (total - ln) if selection == "over" else (ln - total)
    elif market == "AH":
        ln = float(line if line is not None else 0.0)
        margin = hg - ag
        d = (margin + ln) if selection == "home" else (-margin + ln)
    else:
        return 0.0, VOID

    quarter = is_quarter_line(float(line) if line is not None else (ou_line_from_market(market) or 0.0))
    if quarter:
        if d > 0.5 - 1e-9:
            return gain, WON
        if abs(d - 0.25) < 1e-9:
            return round(gain / 2.0, 2), HALF_WON
        if abs(d + 0.25) < 1e-9:
            return round(-stake / 2.0, 2), HALF_LOST
        return -stake, LOST
    if d > 1e-9:
        return gain, WON
    if abs(d) < 1e-9:
        return 0.0, PUSH
    return -stake, LOST


def settle_bool(market: str, selection: str, line: float | None,
                hg: int, ag: int) -> tuple[bool, bool]:
    """Compatibilité : (gagné, push) — sans les demi-gains."""
    pnl, status = settle_pnl(market, selection, line, 1.0, 2.0, hg, ag)
    return status in WIN_STATUSES, status == PUSH
