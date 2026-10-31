"""Métriques d'évaluation : prédiction ET rentabilité."""
from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Qualité de prédiction
# --------------------------------------------------------------------------
def logloss(P: np.ndarray, y: np.ndarray) -> float:
    P = np.clip(np.asarray(P, dtype=float), 1e-12, 1.0)
    return float(-np.log(P[np.arange(len(y)), y]).mean())


def brier(P: np.ndarray, y: np.ndarray) -> float:
    Y = np.eye(P.shape[1])[y]
    return float(((P - Y) ** 2).sum(axis=1).mean())


def accuracy(P: np.ndarray, y: np.ndarray) -> float:
    return float((P.argmax(axis=1) == y).mean())


def rps(P: np.ndarray, y: np.ndarray) -> float:
    """Ranked Probability Score (adapté aux issues ordonnées H>D>A)."""
    Y = np.eye(P.shape[1])[y]
    cp, cy = np.cumsum(P, axis=1), np.cumsum(Y, axis=1)
    return float((((cp - cy) ** 2)[:, :-1].sum(axis=1) / (P.shape[1] - 1)).mean())


def calibration_table(P: np.ndarray, y: np.ndarray, bins: int = 10) -> list[dict]:
    """Fiabilité : probabilité annoncée vs fréquence observée (classe max)."""
    conf = P.max(axis=1)
    pred = P.argmax(axis=1)
    hit = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        mask = (conf >= lo) & (conf < hi)
        if mask.sum() < 15:
            continue
        rows.append({
            "bin": f"{lo:.1f}-{hi:.1f}", "n": int(mask.sum()),
            "predicted": round(float(conf[mask].mean()), 4),
            "observed": round(float(hit[mask].mean()), 4),
            "gap": round(float(hit[mask].mean() - conf[mask].mean()), 4),
        })
    return rows


def summary_predictions(P: np.ndarray, y: np.ndarray) -> dict:
    return {
        "n": int(len(y)),
        "logloss": round(logloss(P, y), 5),
        "brier": round(brier(P, y), 5),
        "rps": round(rps(P, y), 5),
        "accuracy": round(accuracy(P, y), 4),
        "calibration": calibration_table(P, y),
    }


# --------------------------------------------------------------------------
# Rentabilité
# --------------------------------------------------------------------------
def performance(bets: pd.DataFrame, initial_bankroll: float = 1000.0) -> dict:
    """Calcule les KPI d'un portefeuille de paris réglés."""
    settled = bets[bets["status"].isin(["won", "lost", "push"])].copy() if len(bets) else pd.DataFrame()
    if settled.empty:
        return {
            "n_bets": 0, "staked": 0.0, "pnl": 0.0, "roi": 0.0, "yield": 0.0,
            "win_rate": 0.0, "avg_odds": 0.0, "avg_edge": 0.0, "max_drawdown": 0.0,
            "sharpe": 0.0, "bankroll": initial_bankroll, "equity_curve": [],
            "clv_avg": None, "profit_factor": 0.0, "longest_losing_streak": 0,
        }

    settled = settled.sort_values("kickoff")
    pnl = settled["pnl"].astype(float)
    staked = settled["stake"].astype(float)
    equity = initial_bankroll + pnl.cumsum()
    peak = np.maximum.accumulate(np.concatenate([[initial_bankroll], equity.to_numpy()]))
    dd = (peak - np.concatenate([[initial_bankroll], equity.to_numpy()])) / np.maximum(peak, 1e-9)

    wins = settled[settled["status"] == "won"]
    losses = settled[settled["status"] == "lost"]
    streak = longest = 0
    for s in settled["status"]:
        if s == "lost":
            streak += 1
            longest = max(longest, streak)
        elif s == "won":
            streak = 0

    roi = pnl.sum() / staked.sum() if staked.sum() > 0 else 0.0
    per_bet_roi = (pnl / staked.replace(0, np.nan)).dropna()
    sharpe = float(per_bet_roi.mean() / per_bet_roi.std() * np.sqrt(len(per_bet_roi))) if len(per_bet_roi) > 2 and per_bet_roi.std() > 0 else 0.0

    clv = settled["clv"].dropna() if "clv" in settled.columns else pd.Series(dtype=float)

    return {
        "n_bets": int(len(settled)),
        "staked": round(float(staked.sum()), 2),
        "pnl": round(float(pnl.sum()), 2),
        "roi": round(float(roi), 4),
        "yield": round(float(roi), 4),
        "win_rate": round(float((settled["status"] == "won").mean()), 4),
        "avg_odds": round(float(settled["price"].mean()), 3),
        "avg_edge": round(float(settled["edge"].mean()), 4),
        "avg_stake": round(float(staked.mean()), 2),
        "max_drawdown": round(float(dd.max()), 4),
        "sharpe": round(sharpe, 3),
        "bankroll": round(float(equity.iloc[-1]), 2),
        "profit_factor": round(float(wins["pnl"].sum() / abs(losses["pnl"].sum())), 3) if len(losses) and losses["pnl"].sum() != 0 else 0.0,
        "longest_losing_streak": int(longest),
        "clv_avg": round(float(clv.mean()), 4) if len(clv) else None,
        "equity_curve": [
            {"date": str(d)[:10], "bankroll": round(float(v), 2)}
            for d, v in zip(settled["kickoff"], equity, strict=False)
        ][-400:],
    }


def breakdown(bets: pd.DataFrame, by: str, initial_bankroll: float = 1000.0) -> list[dict]:
    if bets.empty or by not in bets.columns:
        return []
    rows = []
    for key, grp in bets.groupby(by):
        p = performance(grp, initial_bankroll)
        rows.append({"group": str(key), "n": p["n_bets"], "roi": p["roi"],
                     "pnl": p["pnl"], "staked": p["staked"], "win_rate": p["win_rate"]})
    return sorted(rows, key=lambda r: -r["n"])
