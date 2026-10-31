"""Ensemble ATHENA : Dixon-Coles + gradient boosting, calibré.

Étapes :
  1. Dixon-Coles par ligue (interprétable, robuste aux petits échantillons)
  2. LightGBM multi-ligues (capte les interactions, la forme, le calendrier)
  3. Mélange convexe au niveau des MATRICES de scores ; le poids est estimé
     par validation temporelle (jamais sur les données de test)
  4. Temperature scaling pour corriger la sur/sous-confiance
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from ..logging_conf import get_logger
from .dixon_coles import DixonColesModel, fit_leagues
from .gbm import GoalModel
from .markets import ScoreMatrix

log = get_logger("model.ensemble")

RESULTS = {0: "home", 1: "draw", 2: "away"}


def _logloss(P: np.ndarray, y: np.ndarray) -> float:
    P = np.clip(P, 1e-9, 1.0)
    return float(-np.log(P[np.arange(len(y)), y]).mean())


@dataclass
class EnsembleModel:
    dc: dict[str, DixonColesModel]
    gbm: GoalModel | None
    w_dc: float = 0.55
    temperature: float = 1.0
    max_goals: int = 10
    version: str = "1.0"
    trained_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metrics: dict = field(default_factory=dict)
    calibrators: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ core
    def matrix(self, league: str, home: str, away: str, feat_row: pd.DataFrame | None = None) -> ScoreMatrix:
        """Matrice de scores finale pour un match."""
        dc_model = self.dc.get(str(league))
        dc_mat = dc_model.matrix(home, away) if dc_model else None

        gbm_mat = None
        if self.gbm is not None and feat_row is not None and len(feat_row) > 0:
            try:
                gbm_mat = self.gbm.predict_matrices(feat_row)[0]
            except Exception as exc:  # noqa: BLE001
                log.warning("GBM indisponible (%s), repli Dixon-Coles", exc)

        if dc_mat is None and gbm_mat is None:
            # repli : ligue inconnue -> Poisson moyenne division
            return ScoreMatrix.from_lambdas(1.45, 1.20, self.max_goals, rho=-0.03, label="prior")
        if gbm_mat is None:
            mat = dc_mat
        elif dc_mat is None:
            mat = gbm_mat
        else:
            mat = dc_mat.blend(gbm_mat, self.w_dc)
        if abs(self.temperature - 1.0) > 1e-6:
            mat = mat.sharpen(self.temperature)
        return mat

    def predict_frame(self, feats: pd.DataFrame, w_dc: float | None = None) -> list[ScoreMatrix]:
        """Prédiction vectorisée (utilisée par le backtest)."""
        if feats.empty:
            return []
        out: list[ScoreMatrix] = []
        gbm_mats: dict[int, ScoreMatrix] = {}
        if self.gbm is not None:
            try:
                for i, m in zip(feats.index, self.gbm.predict_matrices(feats), strict=False):
                    gbm_mats[i] = m
            except Exception as exc:  # noqa: BLE001
                log.warning("GBM en échec sur le lot : %s", exc)
        w = self.w_dc if w_dc is None else w_dc
        for i, row in feats.iterrows():
            dc_model = self.dc.get(str(row.get("league")))
            dc_mat = dc_model.matrix(row["home"], row["away"]) if dc_model else None
            g = gbm_mats.get(i)
            if dc_mat is None and g is None:
                m = ScoreMatrix.from_lambdas(1.45, 1.20, self.max_goals, rho=-0.03, label="prior")
            elif g is None:
                m = dc_mat
            elif dc_mat is None:
                m = g
            else:
                m = dc_mat.blend(g, w)
            if abs(self.temperature - 1.0) > 1e-6:
                m = m.sharpen(self.temperature)
            out.append(m)
        return out

    def to_dict(self) -> dict:
        return {
            "version": self.version, "trained_at": self.trained_at,
            "w_dc": self.w_dc, "temperature": self.temperature,
            "max_goals": self.max_goals, "metrics": self.metrics,
            "dc": {k: v.to_dict() for k, v in self.dc.items()},
            "gbm_importance": self.gbm.feature_importance(18) if self.gbm else {},
            "gbm_metrics": self.gbm.metrics if self.gbm else {},
        }


def optimize_blend(dc_mats: list[ScoreMatrix], gbm_mats: list[ScoreMatrix],
                   y: np.ndarray, grid: np.ndarray | None = None) -> tuple[float, float]:
    """Choisit le poids du Dixon-Coles qui minimise la logloss 1X2."""
    grid = grid if grid is not None else np.arange(0.0, 1.001, 0.05)
    best_w, best_ll = 0.5, float("inf")
    for w in grid:
        P = np.array([[m.blend(g, float(w)).probs_1x2()[k] for k in ("home", "draw", "away")]
                      for m, g in zip(dc_mats, gbm_mats, strict=False)])
        ll = _logloss(P, y)
        if ll < best_ll:
            best_w, best_ll = float(w), ll
    return best_w, best_ll


def optimize_temperature(mats: list[ScoreMatrix], y: np.ndarray) -> tuple[float, float]:
    """Temperature scaling sur la matrice (minimise la logloss 1X2)."""
    def obj(T: float) -> float:
        P = np.array([[m.sharpen(T).probs_1x2()[k] for k in ("home", "draw", "away")] for m in mats])
        return _logloss(P, y)

    res = minimize_scalar(obj, bounds=(0.85, 1.30), method="bounded",
                          options={"xatol": 1e-3})
    return float(res.x), float(res.fun)


def train(cfg, history: pd.DataFrame, train_feats: pd.DataFrame,
          val_feats: pd.DataFrame | None = None) -> EnsembleModel:
    """Entraîne Dixon-Coles (par ligue) + GBM, puis calibre l'ensemble."""
    mc = cfg.model
    ref = pd.Timestamp(train_feats["date"].max()) if len(train_feats) else pd.Timestamp.utcnow()

    dc_models = fit_leagues(
        history[history["date"] <= ref], ref_date=ref,
        xi=float(mc.get("xi_decay_per_day", 0.0019)),
        ridge=float(mc.get("ridge", 0.035)),
        max_goals=int(mc.get("max_goals", 10)),
        train_window_days=int(mc.get("train_window_days", 1500)),
        min_matches_team=int(mc.get("min_matches_team", 4)),
    )

    gbm = None
    if mc.get("gbm", {}).get("enabled", True):
        gbm = GoalModel(params=mc.get("gbm", {}), max_goals=int(mc.get("max_goals", 10)))
        gbm.fit(train_feats, valid=val_feats)
        log.info("GBM entraîné : logloss val=%s", gbm.metrics.get("val_logloss"))

    ens = EnsembleModel(dc=dc_models, gbm=gbm, w_dc=float(mc.get("dc_weight_default", 0.55)),
                        max_goals=int(mc.get("max_goals", 10)))
    ens.version = f"athena-{cfg.raw.get('version', '1.0')}-{pd.Timestamp.utcnow():%Y%m%d}"

    # ---- calibration sur la validation (poids + température) -------------
    if val_feats is not None and len(val_feats) > 100 and gbm is not None:
        v = val_feats.dropna(subset=["result_code", "home_goals", "away_goals"]).copy()
        v["result_code"] = v["result_code"].astype(int)
        y = v["result_code"].to_numpy()
        dc_mats, gbm_mats = [], []
        for _i, row in v.iterrows():
            dc_model = dc_models.get(str(row.get("league")))
            dc_mats.append(dc_model.matrix(row["home"], row["away"]) if dc_model
                           else ScoreMatrix.from_lambdas(1.45, 1.20, ens.max_goals, -0.03))
        gbm_mats = gbm.predict_matrices(v)
        w, ll_w = optimize_blend(dc_mats, gbm_mats, y)
        blended = [d.blend(g, w) for d, g in zip(dc_mats, gbm_mats, strict=False)]
        T, ll_t = optimize_temperature(blended, y)
        ens.w_dc, ens.temperature = w, T
        ens.metrics.update({"val_logloss_blend": round(ll_w, 5), "val_logloss_calibrated": round(ll_t, 5)})
        log.info("calibration : w_dc=%.2f  T=%.3f  logloss %.4f -> %.4f", w, T, ll_w, ll_t)
    return ens
