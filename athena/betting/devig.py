"""Retrait de la marge (de-vigging) : cotes -> probabilités "pures".

Une cote décimale contient la marge du bookmaker. Quatre méthodes :

  - multiplicative : p_i = (1/o_i) / Σ(1/o_j)      (simple, biaisée sur les
    favoris)
  - additive       : p_i = 1/o_i - (Σ-1)/n         (répartit la marge)
  - power          : Σ p_i^k = 1                   (bon compromis, référence)
  - shin           : modèle de Shin (1993), corrige le biais "favorite-longshot"

Le choix par défaut (config) est ``power`` : c'est la méthode qui maximise
la logloss sur nos données historiques tout en restant stable.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

EPS = 1e-9


def implied(odds: list[float]) -> np.ndarray:
    o = np.array(odds, dtype=float)
    o = np.clip(o, 1.001, None)
    return 1.0 / o


def overround(odds: list[float]) -> float:
    return float(implied(odds).sum())


def multiplicative(odds: list[float]) -> np.ndarray:
    p = implied(odds)
    s = p.sum()
    return p / s if s > 0 else p


def additive(odds: list[float]) -> np.ndarray:
    p = implied(odds)
    n = len(p)
    s = p.sum()
    out = p - (s - 1.0) / n
    out = np.clip(out, EPS, None)
    return out / out.sum()


def power(odds: list[float]) -> np.ndarray:
    p = implied(odds)

    def f(k: float) -> float:
        return float(np.power(p, k).sum() - 1.0)

    try:
        lo, hi = 0.2, 5.0
        if f(lo) * f(hi) > 0:
            return multiplicative(odds)
        k = brentq(f, lo, hi, xtol=1e-6)
        out = np.power(p, k)
        return out / out.sum()
    except Exception:  # noqa: BLE001
        return multiplicative(odds)


def shin(odds: list[float]) -> np.ndarray:
    """Shin (1993) : estime la proportion z de parieurs informés."""
    p = implied(odds)

    def probs(z: float) -> np.ndarray:
        return (np.sqrt(z ** 2 + 4 * (1 - z) * p ** 2 / p.sum()) - z) / (2 * (1 - z))

    def g(z: float) -> float:
        return float(probs(z).sum() - 1.0)

    try:
        z = brentq(g, 1e-6, 0.5, xtol=1e-8)
        out = probs(z)
        return out / out.sum()
    except Exception:  # noqa: BLE001
        return multiplicative(odds)


METHODS = {
    "multiplicative": multiplicative,
    "additive": additive,
    "power": power,
    "shin": shin,
}


def devig_named(odds: list[float], names: list[str], method: str = "power") -> dict[str, float]:
    """Dé-vig générique : n issues nommées."""
    if len(odds) != len(names):
        raise ValueError("tailles incohérentes")
    fn = METHODS.get(method, power)
    p = np.clip(fn(odds), EPS, 1.0)
    p = p / p.sum()
    return {n: float(v) for n, v in zip(names, p, strict=False)}


def devig(odds: list[float], method: str = "power") -> dict[str, float]:
    """Cotes [home, draw, away] -> probabilités normalisées."""
    return devig_named(odds, ["home", "draw", "away"], method)


def devig_two(odds: list[float], method: str = "power") -> dict[str, float]:
    """Marché binaire (over/under, BTTS oui/non, handicap)."""
    return devig_named(odds, ["a", "b"], method)


def fair_odds(p: float, margin: float = 0.0) -> float:
    """Probabilité -> cote "juste". ``margin`` simule la marge d'un book
    (elle *baisse* la cote proposée : 0,05 = 5 % de marge)."""
    p = max(min(p, 0.999), 0.001)
    return round(1.0 / (p * (1.0 + margin)), 3)


def blend_with_market(p_model: dict[str, float], p_market: dict[str, float],
                      lambda_market: float = 0.3) -> dict[str, float]:
    """Rapproche le modèle du marché sharp (réduction de variance).

    λ_market = 0 -> modèle pur ; λ_market = 1 -> marché pur.
    Un λ intermédiaire (0.25-0.40) est ce qui produit les edges les plus
    stables en pratique : le marché apporte l'information que le modèle
    n'a pas (compositions, blessures, argent intelligent).
    """
    lam = float(np.clip(lambda_market, 0.0, 1.0))
    keys = set(p_model) & set(p_market)
    out = {k: (1 - lam) * p_model[k] + lam * p_market[k] for k in keys}
    s = sum(out.values())
    return {k: v / s for k, v in out.items()} if s > 0 else dict(p_model)
