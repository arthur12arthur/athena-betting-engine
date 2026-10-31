"""Gestion de bankroll : Kelly fractionnaire avec garde-fous.

Kelly brut : f* = (p·b - q) / b  avec b = cote - 1.
Un Kelly plein est trop volatil et suppose que p est exact -> on applique :
  - une fraction (22 % par défaut) : robustesse à l'erreur d'estimation
  - un plafond par pari (2 % de bankroll)
  - un plafond d'exposition journalière (7 %)
  - un plancher de mise (une mise < 0,1 % n'a pas de sens)
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StakePlan:
    stake: float
    kelly_full: float
    kelly_used: float
    capped_by: str = ""

    def to_dict(self) -> dict:
        return {"stake": round(self.stake, 2), "kelly_full": round(self.kelly_full, 4),
                "kelly_used": round(self.kelly_used, 4), "capped_by": self.capped_by}


def kelly_fraction(p: float, odds: float) -> float:
    """Fraction de bankroll Kelly optimale (0 si pas d'edge)."""
    if odds <= 1.0 or not (0.0 < p < 1.0):
        return 0.0
    b = odds - 1.0
    f = (p * b - (1.0 - p)) / b
    return max(0.0, float(f))


def plan_stake(p: float, odds: float, bankroll: float, cfg: dict,
               remaining_exposure: float | None = None) -> StakePlan:
    """Mise finale pour un pari, tous garde-fous appliqués."""
    frac = float(cfg.get("kelly_fraction", 0.22))
    max_pct = float(cfg.get("max_stake_pct", 0.02))
    min_pct = float(cfg.get("min_stake_pct", 0.001))

    f = kelly_fraction(p, odds)
    stake = f * frac * bankroll
    capped = ""
    if stake > max_pct * bankroll:
        stake = max_pct * bankroll
        capped = "max_stake_pct"
    if remaining_exposure is not None and stake > remaining_exposure:
        stake = max(0.0, remaining_exposure)
        capped = "daily_exposure"
    if stake < min_pct * bankroll:
        stake = 0.0
        capped = "below_min"
    return StakePlan(stake=round(stake, 2), kelly_full=round(f, 4),
                     kelly_used=round(stake / bankroll, 5) if bankroll else 0.0, capped_by=capped)


def settle_bet(stake: float, odds: float, won: bool, push: bool = False) -> float:
    """Profit net d'un pari (push = remboursé sur handicap entier)."""
    if push:
        return 0.0
    return round(stake * (odds - 1.0), 2) if won else round(-stake, 2)
