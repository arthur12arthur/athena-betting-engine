"""Modèle Dixon-Coles (Poisson bivarié avec décroissance temporelle).

Référence : Dixon & Coles (1997), "Modelling Association Football Scores and
Inefficiencies in the Football Betting Market".

    log λ_dom = μ + γ + att[dom] - def[ext]
    log λ_ext = μ     + att[ext] - def[dom]

  - γ   : avantage du terrain
  - att/def : forces offensives/défensives estimées par maximum de
    vraisemblance pondéré (les matchs anciens pèsent moins : exp(-ξ·Δt))
  - τ   : correction des scores faibles (0-0, 1-0, 0-1, 1-1)
  - ridge : pénalité L2 sur att/def (indispensable pour les promus, qui
    n'ont pas encore d'historique dans la division)

Le modèle est ré-estimé par ligue et par fenêtre glissante : aucune
information future n'entre dans l'estimation (backtest honnête).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..logging_conf import get_logger
from .markets import ScoreMatrix

log = get_logger("model.dc")

EPS = 1e-12


@dataclass
class DixonColesModel:
    teams: list[str]
    att: dict[str, float]
    deff: dict[str, float]
    mu: float
    home_adv: float
    rho: float
    max_goals: int = 10
    n_matches: int = 0
    ref_date: str | None = None
    xi: float = 0.0019
    optim: dict = field(default_factory=dict)

    # ------------------------------------------------------------- prédiction
    def lambdas(self, home: str, away: str) -> tuple[float, float]:
        ah = self.att.get(home, 0.0)
        dh = self.deff.get(home, 0.0)
        aa = self.att.get(away, 0.0)
        da = self.deff.get(away, 0.0)
        lh = np.exp(self.mu + self.home_adv + ah - da)
        la = np.exp(self.mu + aa - dh)
        return float(np.clip(lh, 0.05, 8.0)), float(np.clip(la, 0.05, 8.0))

    def matrix(self, home: str, away: str) -> ScoreMatrix:
        lh, la = self.lambdas(home, away)
        return ScoreMatrix.from_lambdas(lh, la, self.max_goals, self.rho, label="dixon_coles")

    def knows(self, team: str, min_matches: int = 3) -> bool:
        return team in self.att and self.att.get(team, 0.0) != 0.0 or min_matches <= 0

    def team_strength(self) -> pd.DataFrame:
        rows = [{"team": t, "attack": self.att.get(t, 0.0), "defense": self.deff.get(t, 0.0)}
                for t in self.teams]
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        # convention : `attack` et `defense` croissent tous deux avec la
        # qualité (une défense forte a un coefficient élevé, car il est
        # retranché aux buts adverses).
        df["rating"] = df["attack"] + df["defense"]
        return df.sort_values("rating", ascending=False).reset_index(drop=True)

    def to_dict(self) -> dict:
        return {
            "type": "dixon_coles", "mu": round(self.mu, 4), "home_adv": round(self.home_adv, 4),
            "rho": round(self.rho, 4), "n_matches": self.n_matches, "ref_date": self.ref_date,
            "xi": self.xi, "teams": self.teams,
            "att": {k: round(v, 4) for k, v in self.att.items()},
            "deff": {k: round(v, 4) for k, v in self.deff.items()},
        }

    @classmethod
    def from_dict(cls, d: dict, max_goals: int = 10) -> DixonColesModel:
        return cls(teams=d["teams"], att=d["att"], deff=d["deff"], mu=d["mu"],
                   home_adv=d["home_adv"], rho=d["rho"], max_goals=max_goals,
                   n_matches=d.get("n_matches", 0), ref_date=d.get("ref_date"),
                   xi=d.get("xi", 0.0019))

    # ------------------------------------------------------------------- fit
    @classmethod
    def fit(cls, df: pd.DataFrame, ref_date: datetime | pd.Timestamp | None = None,
            xi: float = 0.0019, ridge: float = 0.035, max_goals: int = 10,
            min_matches_team: int = 4) -> DixonColesModel | None:
        """Estime le modèle sur un DataFrame de matchs terminés."""
        d = df.dropna(subset=["home_goals", "away_goals"]).copy()
        if len(d) < 60:
            log.warning("Dixon-Coles: pas assez de matchs (%d)", len(d))
            return None
        ref = pd.Timestamp(ref_date) if ref_date is not None else pd.Timestamp(d["date"].max())

        counts = pd.concat([d["home"], d["away"]]).value_counts()
        keep = set(counts[counts >= min_matches_team].index)
        d = d[d["home"].isin(keep) & d["away"].isin(keep)]
        if len(d) < 60:
            return None

        teams = sorted(set(d["home"]) | set(d["away"]))
        tidx = {t: i for i, t in enumerate(teams)}
        n = len(teams)

        hg = d["home_goals"].to_numpy(dtype=float)
        ag = d["away_goals"].to_numpy(dtype=float)
        ih = d["home"].map(tidx).to_numpy(dtype=int)
        ia = d["away"].map(tidx).to_numpy(dtype=int)

        days_ago = (ref - pd.to_datetime(d["date"])).dt.total_seconds().to_numpy() / 86400.0
        days_ago = np.clip(days_ago, 0.0, None)
        w = np.exp(-xi * days_ago)
        w = w / w.mean()

        def unpack(theta: np.ndarray):
            return theta[:n], theta[n:2 * n], theta[2 * n], theta[2 * n + 1], theta[2 * n + 2]

        def nll(theta: np.ndarray) -> float:
            att, deff, mu, hfa, rho = unpack(theta)
            lh = np.exp(np.clip(mu + hfa + att[ih] - deff[ia], -6, 4))
            la = np.exp(np.clip(mu + att[ia] - deff[ih], -6, 4))
            tau = np.ones_like(lh)
            m00 = (hg == 0) & (ag == 0)
            m01 = (hg == 0) & (ag == 1)
            m10 = (hg == 1) & (ag == 0)
            m11 = (hg == 1) & (ag == 1)
            tau = np.where(m00, 1.0 - lh * la * rho, tau)
            tau = np.where(m01, 1.0 + lh * rho, tau)
            tau = np.where(m10, 1.0 + la * rho, tau)
            tau = np.where(m11, 1.0 - rho, tau)
            tau = np.clip(tau, 1e-9, None)
            ll = -lh - la + hg * np.log(lh) + ag * np.log(la) + np.log(tau)
            pen = ridge * float(att @ att + deff @ deff) + 4.0 * rho * rho
            return float(-(w * ll).sum() / w.sum() + pen)

        x0 = np.concatenate([np.zeros(2 * n), [np.log(max(d[["home_goals"]].mean().item(), 0.2)) - 0.15,
                                               0.25, -0.03]])
        bounds = [(-3, 3)] * (2 * n) + [(-3, 3), (-1.0, 1.0), (-0.35, 0.35)]
        res = minimize(nll, x0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 400, "ftol": 1e-10})
        att, deff, mu, hfa, rho = unpack(res.x)
        # contrainte d'identification : moyenne des attaques = 0
        shift = float(np.mean(att))
        att = att - shift

        model = cls(
            teams=teams,
            att={t: float(att[i]) for t, i in tidx.items()},
            deff={t: float(deff[i]) for t, i in tidx.items()},
            mu=float(mu + shift), home_adv=float(hfa), rho=float(rho),
            max_goals=max_goals, n_matches=int(len(d)),
            ref_date=str(pd.Timestamp(ref).date()), xi=xi,
            optim={"success": bool(res.success), "nll": round(float(res.fun), 5),
                   "nit": int(res.nit), "nfev": int(res.nfev)},
        )
        return model


def fit_leagues(history: pd.DataFrame, ref_date: datetime | pd.Timestamp | None = None,
                xi: float = 0.0019, ridge: float = 0.035, max_goals: int = 10,
                train_window_days: int = 1500, min_matches_team: int = 4) -> dict[str, DixonColesModel]:
    """Estime un modèle Dixon-Coles par ligue sur une fenêtre glissante."""
    ref = pd.Timestamp(ref_date) if ref_date is not None else pd.Timestamp(history["date"].max())
    window_start = ref - pd.Timedelta(days=train_window_days)
    out: dict[str, DixonColesModel] = {}
    for league, grp in history.groupby("league"):
        sub = grp[(grp["date"] <= ref) & (grp["date"] >= window_start)]
        model = DixonColesModel.fit(sub, ref_date=ref, xi=xi, ridge=ridge,
                                    max_goals=max_goals, min_matches_team=min_matches_team)
        if model is not None:
            out[str(league)] = model
            log.info("DC %-4s: %3d équipes, %4d matchs, γ=%.2f, ρ=%+.3f, nll=%.4f",
                     league, len(model.teams), model.n_matches, model.home_adv,
                     model.rho, model.optim.get("nll", float("nan")))
    return out
