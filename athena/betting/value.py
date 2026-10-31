"""Détection de valeur : modèle vs marché, edge et cotes justes.

Principe (value betting) :
  1. on estime la probabilité réelle p (modèle + ancrage marché sharp)
  2. on compare à la cote proposée par le bookmaker : EV = p·cote - 1
  3. on ne parie que si l'edge dépasse un seuil ET si le marché est liquide

Les edges ne sont pas "gratuits" : la marge de nos estimations (~1 %) impose
des seuils prudents. Un edge de 2,5 % contre le marché moyen, vérifié sur
des dizaines de milliers de paris, est déjà un très bon système.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..logging_conf import get_logger
from .devig import blend_with_market

log = get_logger("betting.value")


@dataclass
class ValueBet:
    market: str
    selection: str
    line: float | None
    price: float
    p_model: float
    p_fair: float
    p_market: float
    edge: float
    ev: float
    kelly: float
    fair_odds: float

    def to_dict(self) -> dict:
        return {
            "market": self.market, "selection": self.selection, "line": self.line,
            "price": round(self.price, 3), "p_model": round(self.p_model, 4),
            "p_fair": round(self.p_fair, 4), "p_market": round(self.p_market, 4),
            "edge": round(self.edge, 4), "ev": round(self.ev, 4),
            "kelly": round(self.kelly, 4), "fair_odds": self.fair_odds,
        }


SELECTION_LABELS = {
    ("1X2", "home"): "1 (domicile)", ("1X2", "draw"): "X (nul)", ("1X2", "away"): "2 (extérieur)",
    ("DC", "1X"): "1X", ("DC", "12"): "12", ("DC", "X2"): "X2",
    ("OU25", "over"): "Plus de 2,5 buts", ("OU25", "under"): "Moins de 2,5 buts",
    ("OU", "over"): "Plus", ("OU", "under"): "Moins",
    ("BTTS", "yes"): "Les 2 équipes marquent", ("BTTS", "no"): "Pas les 2 équipes",
}


def label(market: str, selection: str, line: float | None = None) -> str:
    if (market, selection) in SELECTION_LABELS:
        return SELECTION_LABELS[(market, selection)]
    if market == "AH":
        return f"Handicap {selection} {line:+g}"
    return f"{market} {selection}"


def evaluate_market(market: str, p_model: dict[str, float], p_market: dict[str, float],
                    odds: dict[str, float], cfg: dict, lambda_market: float = 0.3,
                    sharp: bool = False) -> list[ValueBet]:
    """Évalue toutes les sélections d'un marché et renvoie celles qui ont de la valeur."""
    p_fair = blend_with_market(p_model, p_market, lambda_market)
    out: list[ValueBet] = []
    min_edge = float(cfg.get("min_edge_sharp" if sharp else "min_edge", 0.025))
    for sel, price in odds.items():
        if sel not in p_fair or not price or price <= 1.0:
            continue
        p = float(p_fair[sel])
        if not (float(cfg.get("min_prob", 0.05)) <= p <= float(cfg.get("max_prob", 0.85))):
            continue
        if not (float(cfg.get("min_odds", 1.45)) <= price <= float(cfg.get("max_odds", 8.0))):
            continue
        edge = p * price - 1.0
        if edge < min_edge:
            continue
        out.append(ValueBet(
            market=market, selection=sel, line=None, price=float(price),
            p_model=float(p_model.get(sel, 0.0)), p_fair=p,
            p_market=float(p_market.get(sel, 0.0)), edge=float(edge), ev=float(edge),
            kelly=float(max(0.0, (p * (price - 1) - (1 - p)) / (price - 1))),
            fair_odds=round(1.0 / max(p, 1e-6), 3),
        ))
    return out


def markets_for_matrix(matrix, markets: list[str]) -> dict[str, dict[str, float]]:
    """Extrait les marchés activés depuis une matrice de scores."""
    out: dict[str, dict[str, float]] = {}
    for m in markets:
        try:
            out[m] = matrix.market(m)
        except Exception as exc:  # noqa: BLE001
            log.debug("marché %s indisponible: %s", m, exc)
    return out
