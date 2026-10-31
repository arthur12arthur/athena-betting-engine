"""Modèle gradient boosting (LightGBM) : probabilités 1X2 + buts attendus.

Deux têtes :
  1) classifieur multiclasse (H/D/A)  -> probabilités calibrées par le modèle
  2) deux régresseurs Poisson         -> λ_domicile, λ_extérieur

La matrice de scores est reconstruite à partir des λ puis recalée sur les
probabilités 1X2 du classifieur : le marché 1X2 et les marchés de buts
restent cohérents tout en bénéficiant chacun du meilleur sous-modèle.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..etl.features import CATEGORICAL, FEATURES
from ..logging_conf import get_logger
from .markets import ScoreMatrix

log = get_logger("model.gbm")


class GoalModel:
    def __init__(self, params: dict, max_goals: int = 10, decay: float = 0.0008):
        self.params = params
        self.max_goals = max_goals
        self.decay = decay
        self.clf: lgb.LGBMClassifier | None = None
        self.reg_home: lgb.LGBMRegressor | None = None
        self.reg_away: lgb.LGBMRegressor | None = None
        self.medians: dict[str, float] = {}
        self.feature_names: list[str] = FEATURES
        self.metrics: dict = {}

    # ------------------------------------------------------------------ fit
    def _matrix(self, df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        X = df[self.feature_names].copy()
        if self.medians:
            X = X.fillna(self.medians)
        X = X.fillna(0.0)
        # IMPORTANT : on garde des entiers (jamais de dtype "category") pour que
        # les codes d'entraînement et de prédiction coïncident. LightGBM traite
        # la colonne comme catégorielle via `categorical_feature`.
        for c in CATEGORICAL:
            if c in X.columns:
                X[c] = X[c].astype("int32")
        w = None
        if "date" in df.columns and self.decay > 0:
            days = (pd.to_datetime(df["date"]).max() - pd.to_datetime(df["date"])).dt.days
            w = np.exp(-self.decay * days.to_numpy(dtype=float))
        return X, w

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None) -> GoalModel:
        d = train.dropna(subset=["result_code", "home_goals", "away_goals"]).copy()
        d["result_code"] = d["result_code"].astype(int)
        self.medians = {c: float(d[c].median()) for c in self.feature_names if c in d.columns}
        self.medians = {k: (0.0 if np.isnan(v) else v) for k, v in self.medians.items()}

        X, w = self._matrix(d)
        y = d["result_code"].to_numpy()

        p = dict(self.params)
        common = dict(
            n_estimators=p.get("n_estimators", 450),
            learning_rate=p.get("learning_rate", 0.03),
            num_leaves=p.get("num_leaves", 24),
            min_child_samples=p.get("min_child_samples", 35),
            subsample=p.get("subsample", 0.85),
            subsample_freq=1,
            colsample_bytree=p.get("colsample_bytree", 0.8),
            reg_lambda=p.get("reg_lambda", 1.0),
            n_jobs=2,
            verbose=-1,
            random_state=42,
        )
        self.clf = lgb.LGBMClassifier(objective="multiclass", num_class=3, **common)
        self.clf.fit(X, y, sample_weight=w,
                     categorical_feature=[c for c in CATEGORICAL if c in X.columns])

        self.reg_home = lgb.LGBMRegressor(objective="poisson", **common)
        self.reg_home.fit(X, d["home_goals"].to_numpy(dtype=float), sample_weight=w,
                          categorical_feature=[c for c in CATEGORICAL if c in X.columns])
        self.reg_away = lgb.LGBMRegressor(objective="poisson", **common)
        self.reg_away.fit(X, d["away_goals"].to_numpy(dtype=float), sample_weight=w,
                          categorical_feature=[c for c in CATEGORICAL if c in X.columns])

        if valid is not None and len(valid) > 30:
            from sklearn.metrics import log_loss
            P = self.predict_proba(valid)
            yv = valid["result_code"].dropna().astype(int).to_numpy()
            if len(yv) == len(P):
                self.metrics["val_logloss"] = float(log_loss(yv, P, labels=[0, 1, 2]))
                self.metrics["val_acc"] = float((P.argmax(1) == yv).mean())
        return self

    # -------------------------------------------------------------- predict
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X, _ = self._matrix(df)
        return self.clf.predict_proba(X)  # type: ignore[union-attr]

    def predict_lambdas(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X, _ = self._matrix(df)
        lh = np.clip(self.reg_home.predict(X), 0.05, 8.0)  # type: ignore[union-attr]
        la = np.clip(self.reg_away.predict(X), 0.05, 8.0)  # type: ignore[union-attr]
        return lh, la

    def predict_matrices(self, df: pd.DataFrame) -> list[ScoreMatrix]:
        if df.empty:
            return []
        P = self.predict_proba(df)
        lh, la = self.predict_lambdas(df)
        out = []
        for i in range(len(df)):
            m = ScoreMatrix.from_lambdas(lh[i], la[i], self.max_goals, label="gbm")
            out.append(m.reconcile_to(P[i, 0], P[i, 1], P[i, 2]))
        return out

    # ------------------------------------------------------------ persistance
    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.clf.booster_.save_model(str(path / "clf.txt"))          # type: ignore[union-attr]
        self.reg_home.booster_.save_model(str(path / "reg_home.txt"))  # type: ignore[union-attr]
        self.reg_away.booster_.save_model(str(path / "reg_away.txt"))  # type: ignore[union-attr]
        (path / "meta.json").write_text(json.dumps(
            {"medians": self.medians, "feature_names": self.feature_names,
             "max_goals": self.max_goals, "metrics": self.metrics, "params": self.params},
            indent=1, default=float), encoding="utf-8")

    def feature_importance(self, top: int = 15) -> dict[str, float]:
        if self.clf is None:
            return {}
        imp = self.clf.feature_importances_
        names = self.clf.feature_name_
        pairs = sorted(zip(names, imp, strict=False), key=lambda x: -x[1])[:top]
        total = float(sum(i for _, i in pairs)) or 1.0
        return {n: round(i / total, 4) for n, i in pairs}
