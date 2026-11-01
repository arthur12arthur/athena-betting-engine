"""Matrices de scores -> probabilités de marchés.

Tout le système raisonne sur une **matrice de scores** P(i, j) = probabilité
que le match se termine sur le score exact (i, j). C'est ce qui garantit la
cohérence entre marchés : 1X2, double chance, over/under, BTTS et handicaps
sont tous dérivés de la même distribution.

Fonctions clés :
  - ``ScoreMatrix.from_lambdas``   : Poisson indépendant ou bivarié Dixon-Coles
  - ``ScoreMatrix.reconcile_to``   : recalage de la matrice sur des
    probabilités 1X2 cibles (utilisé pour l'ensemble et la calibration)
  - ``ScoreMatrix.blend``          : mélange convexe de deux matrices
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import poisson

EPS = 1e-12


def is_integer_line(line: float) -> bool:
    """Ligne entière -> remboursement (push) possible."""
    return abs(float(line) - round(float(line))) < 1e-9


def _is_quarter_line(line: float) -> bool:
    """Ligne quart : 2*line est un demi-entier (ex. -0,75 -> -1,5)."""
    x = float(line) * 2.0
    return abs(x - round(x)) > 1e-9


def is_quarter_line(line: float) -> bool:
    return _is_quarter_line(line)


def ou_line_from_market(market: str) -> float | None:
    """Déduit la ligne d'un libellé de marché.

    'OU25' -> 2,5 (notation historique condensée) · 'OU2.5' -> 2,5 · 'OU3' -> 3,0
    """
    if not market:
        return None
    m = market.upper()
    if not m.startswith("OU") or len(m) <= 2:
        return None
    suffix = m[2:]
    try:
        if "." in suffix:
            return float(suffix)
        if len(suffix) == 2:
            return float(f"{suffix[0]}.{suffix[1]}")
        if len(suffix) == 1:
            return float(suffix)
        return float(suffix)
    except ValueError:
        return None


def parse_market(market: str, line: float | None = None) -> tuple[str, float | None]:
    """'OU2.5' -> ('OU', 2.5) ; 'AH' -> ('AH', line) ; '1X2' -> ('1X2', None)."""
    m = (market or "").upper()
    if m.startswith("OU"):
        # une ligne explicite prime sur le libellé (ex. marché "OU25" coté à 2,25)
        return "OU", (float(line) if line is not None
                      else ScoreMatrix._ou_line(market, None))
    if m == "AH":
        return "AH", line
    return m, line


def _tau(hg: np.ndarray, ag: np.ndarray, lh: np.ndarray, la: np.ndarray, rho: float) -> np.ndarray:
    """Correction Dixon-Coles des scores faibles (0-0, 1-0, 0-1, 1-1)."""
    t = np.ones_like(lh, dtype=float)
    t = np.where((hg == 0) & (ag == 0), 1.0 - lh * la * rho, t)
    t = np.where((hg == 0) & (ag == 1), 1.0 + lh * rho, t)
    t = np.where((hg == 1) & (ag == 0), 1.0 + la * rho, t)
    t = np.where((hg == 1) & (ag == 1), 1.0 - rho, t)
    return np.clip(t, 1e-9, None)


@dataclass
class ScoreMatrix:
    """Distribution conjointe des scores (home, away)."""

    mat: np.ndarray           # (K, K) lignes = buts domicile
    lam_home: float = 0.0
    lam_away: float = 0.0
    label: str = "model"

    # ------------------------------------------------------------------ init
    @classmethod
    def from_lambdas(cls, lam_home: float, lam_away: float, max_goals: int = 10,
                     rho: float = 0.0, label: str = "poisson") -> ScoreMatrix:
        lam_home = float(np.clip(lam_home, 0.05, 8.0))
        lam_away = float(np.clip(lam_away, 0.05, 8.0))
        k = np.arange(max_goals + 1)
        ph = poisson.pmf(k, lam_home)
        pa = poisson.pmf(k, lam_away)
        mat = np.outer(ph, pa)
        if rho:
            i, j = np.meshgrid(k, k, indexing="ij")
            mat = mat * _tau(i, j, lam_home * np.ones_like(i, dtype=float),
                             lam_away * np.ones_like(j, dtype=float), rho)
            mat = np.clip(mat, 0.0, None)
        mat = mat / mat.sum()
        return cls(mat=mat, lam_home=lam_home, lam_away=lam_away, label=label)

    # ------------------------------------------------------------------ base
    def copy(self) -> ScoreMatrix:
        return ScoreMatrix(self.mat.copy(), self.lam_home, self.lam_away, self.label)

    def probs_1x2(self) -> dict[str, float]:
        home = float(np.tril(self.mat, -1).sum())   # i > j
        draw = float(np.trace(self.mat))
        away = float(np.triu(self.mat, 1).sum())    # j > i
        return {"home": home, "draw": draw, "away": away}

    # -------------------------------------------------------------- recalage
    def reconcile_to(self, p_home: float, p_draw: float, p_away: float) -> ScoreMatrix:
        """Recale la matrice pour que ses blocs 1X2 valent exactement la cible.

        On conserve la *forme* du conditionnel (répartition des scores à
        l'intérieur de chaque issue) et on ne rééchelonne que les trois blocs.
        """
        cur = self.probs_1x2()
        mat = self.mat.copy()
        idx = np.arange(mat.shape[0])
        II, JJ = np.meshgrid(idx, idx, indexing="ij")
        home_mask, draw_mask, away_mask = II > JJ, II == JJ, JJ > II
        targets = {"home": p_home, "draw": p_draw, "away": p_away}
        total = sum(targets.values())
        if total <= 0:
            return self.copy()
        for key, mask in (("home", home_mask), ("draw", draw_mask), ("away", away_mask)):
            mass = cur[key]
            tgt = targets[key] / total
            if mass > EPS:
                mat[mask] *= tgt / mass
            else:  # bloc vide : on répartit uniformément dans le bloc
                n = int(mask.sum())
                if n:
                    mat[mask] = tgt / n
        mat = np.clip(mat, 0.0, None)
        mat = mat / mat.sum()
        return ScoreMatrix(mat=mat, lam_home=self.lam_home, lam_away=self.lam_away, label=self.label)

    def blend(self, other: ScoreMatrix, w: float) -> ScoreMatrix:
        """Mélange convexe : w * self + (1 - w) * other."""
        w = float(np.clip(w, 0.0, 1.0))
        mat = w * self.mat + (1.0 - w) * other.mat
        mat = mat / mat.sum()
        return ScoreMatrix(mat=mat, lam_home=self.lam_home, lam_away=self.lam_away, label="blend")

    def sharpen(self, temperature: float) -> ScoreMatrix:
        """Temperature scaling appliqué à la distribution 1X2.

        T < 1 = plus confiant (écrase la masse sur l'issue favorite),
        T > 1 = plus prudent. On garde la structure conditionnelle de la
        matrice (répartition des scores au sein de chaque issue) et on ne
        rééchelonne que les trois issues, pour rester cohérent avec les
        marchés dérivés.
        """
        T = float(np.clip(temperature, 0.5, 2.5))
        if abs(T - 1.0) < 1e-6:
            return self.copy()
        p = self.probs_1x2()
        q = {k: float(v) ** (1.0 / T) for k, v in p.items()}
        total = sum(q.values())
        q = {k: v / total for k, v in q.items()}
        out = self.reconcile_to(q["home"], q["draw"], q["away"])
        out.label = self.label
        return out

    # -------------------------------------------------------------- marchés
    def market(self, market: str, line: float | None = None) -> dict[str, float]:
        """Probabilités d'un marché. Clés renvoyées = sélections canoniques."""
        idx = np.arange(self.mat.shape[0])
        II, JJ = np.meshgrid(idx, idx, indexing="ij")
        m = self.mat
        if market == "1X2":
            return self.probs_1x2()
        if market == "DC":  # double chance
            p = self.probs_1x2()
            return {"1X": p["home"] + p["draw"], "12": p["home"] + p["away"], "X2": p["draw"] + p["away"]}
        if market == "BTTS":
            p_no = float(m[II == 0].sum() + m[JJ == 0].sum() - m[(II == 0) & (JJ == 0)].sum())
            return {"yes": 1.0 - p_no, "no": p_no}
        if market.startswith("OU"):
            ln = self._ou_line(market, line)
            over = float(m[(II + JJ) > ln].sum())
            draw = float(m[np.abs((II + JJ) - ln) < 1e-9].sum())
            # ligne entière = remboursement possible -> probabilité "équitable"
            return {"over": over, "under": 1.0 - over - draw, "push": draw,
                    "p_over": over + 0.5 * draw, "p_under": 1.0 - over - 0.5 * draw}
        if market == "AH":  # handicap asiatique, ligne exprimée côté domicile
            return self.handicap(-0.5 if line is None else float(line))
        if market == "CS":
            out: dict[str, float] = {}
            for i in range(min(5, self.mat.shape[0])):
                for j in range(min(5, self.mat.shape[1])):
                    out[f"{i}-{j}"] = float(m[i, j])
            return out
        raise ValueError(f"marché inconnu: {market}")


    # ------------------------------------------------------- lignes & handicap
    @staticmethod
    def _ou_line(market: str, line: float | None) -> float:
        """'OU25' -> 2,5 ; 'OU1.5' -> 1,5 ; 'OU' -> paramètre ``line``."""
        ln = ou_line_from_market(market)
        if ln is not None:
            return ln
        return 2.5 if line is None else float(line)

    def handicap(self, line: float) -> dict[str, float]:
        """Probabilités d'un handicap asiatique, lignes entières, demi et **quart**.

        Une ligne quart (ex. -0,75) est la moyenne de deux demi-lignes (-1,0
        et -0,5) : une moitié de mise est poussée, l'autre gagnée/perdue.
        On renvoie donc les probabilités "ajustées du push" (`p_home`, `p_away`),
        la probabilité de push (`push`) et les probabilités de gain pur.
        """
        idx = np.arange(self.mat.shape[0])
        i, j = np.meshgrid(idx, idx, indexing="ij")
        m = self.mat
        margin = i - j

        def one(ln: float) -> tuple[float, float, float]:
            d = margin + ln
            win = float(m[d > 1e-9].sum())
            push = float(m[np.abs(d) < 1e-9].sum())
            lose = float(m[d < -1e-9].sum())
            return win, push, lose

        if _is_quarter_line(line):
            w1, p1, l1 = one(line - 0.25)
            w2, p2, l2 = one(line + 0.25)
            win_home = 0.5 * (w1 + w2)
            push = 0.5 * (p1 + p2)
            lose_home = 0.5 * (l1 + l2)
        else:
            win_home, push, lose_home = one(line)
        p_home = win_home + 0.5 * push
        return {
            "home": p_home, "away": 1.0 - p_home,
            "push": push, "win_home": win_home, "win_away": lose_home,
            "p_home": p_home, "p_away": 1.0 - p_home,
        }

    # --------------------------------------------------------------- utilitaires
    def top_scores(self, n: int = 3) -> list[tuple[str, float]]:
        flat = [(f"{i}-{j}", float(self.mat[i, j]))
                for i in range(self.mat.shape[0]) for j in range(self.mat.shape[1])]
        flat.sort(key=lambda x: -x[1])
        return flat[:n]

    def expected_goals(self) -> dict[str, float]:
        idx = np.arange(self.mat.shape[0])
        ii, jj = np.meshgrid(idx, idx, indexing="ij")
        return {"home": float((self.mat * ii).sum()), "away": float((self.mat * jj).sum())}

    def to_dict(self) -> dict:
        return {
            "lambda_home": round(self.lam_home, 3),
            "lambda_away": round(self.lam_away, 3),
            "p_1x2": {k: round(v, 4) for k, v in self.probs_1x2().items()},
            "p_ou25": {k: round(v, 4) for k, v in self.market("OU25").items()},
            "p_btts": {k: round(v, 4) for k, v in self.market("BTTS").items()},
            "p_dc": {k: round(v, 4) for k, v in self.market("DC").items()},
            "top_scores": [(s, round(p, 4)) for s, p in self.top_scores(3)],
        }
