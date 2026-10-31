"""Cotes multi-bookmakers via The Odds API (palier gratuit).

Optionnel : nécessite une clé gratuite (2 000 requêtes/mois, 500 au départ)
dans la variable d'environnement ``ODDS_API_KEY``. Sans clé, ATHENA
fonctionne avec Pinnacle seul (mode dégradé, moins d'opportunités).

Pourquoi c'est important : le "value betting" vit de l'écart entre la cote
proposée (book soft, souvent en retard) et le prix juste (ligne sharp).
Avec une seule source de cotes, cet écart est nul par construction.
"""
from __future__ import annotations

import os

from ..http import HttpClient
from ..logging_conf import get_logger

log = get_logger("source.oddsapi")

BASE = "https://api.the-odds-api.com/v4"

SPORT_KEYS = {
    "E0": "soccer_epl",
    "E1": "soccer_efl_champ",
    "SP1": "soccer_spain_la_liga",
    "I1": "soccer_italy_serie_a",
    "D1": "soccer_germany_bundesliga",
    "F1": "soccer_france_ligue_one",
    "N1": "soccer_netherlands_eredivisie",
    "B1": "soccer_belgium_first_div",
    "P1": "soccer_portugal_primeira_liga",
    "T1": "soccer_turkey_super_league",
    "SC0": "soccer_spl",
    "G1": "soccer_greece_super_league",
}


class OddsApiClient:
    def __init__(self, key: str | None = None, client: HttpClient | None = None,
                 regions: str = "eu", markets: str = "h2h,totals"):
        self.key = key or os.getenv("ODDS_API_KEY", "").strip()
        self.http = client or HttpClient()
        self.regions = regions
        self.markets = markets
        self.calls = 0
        self.remaining: int | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.key)

    def _get(self, path: str, params: dict | None = None) -> object | None:
        if not self.enabled:
            return None
        p = dict(params or {})
        p["apiKey"] = self.key
        self.calls += 1
        data = self.http.get(f"{BASE}/{path}", params=p, cache_hours=1.0)
        return data

    def odds_for_league(self, league_code: str) -> list[dict]:
        """Renvoie les matchs avec la meilleure cote par sélection."""
        sport = SPORT_KEYS.get(league_code)
        if not sport or not self.enabled:
            return []
        raw = self._get(f"sports/{sport}/odds", {
            "regions": self.regions, "markets": self.markets,
            "oddsFormat": "decimal", "dateFormat": "iso",
        })
        if not isinstance(raw, list):
            return []
        out = []
        for ev in raw:
            best: dict[str, float] = {}
            books: dict[str, str] = {}
            totals: dict[float, dict[str, float]] = {}
            tot_books: dict[tuple, str] = {}
            for bk in ev.get("bookmakers", []):
                for mk in bk.get("markets", []):
                    if mk.get("key") == "h2h":
                        for oc in mk.get("outcomes", []):
                            name = (oc.get("name") or "").lower()
                            price = oc.get("price")
                            key = ("home" if name == ev.get("home_team", "").lower()
                                   else "away" if name == ev.get("away_team", "").lower()
                                   else "draw")
                            if price and (key not in best or price > best[key]):
                                best[key] = float(price)
                                books[key] = bk.get("title", bk.get("key", "?"))
                    elif mk.get("key") == "totals":
                        for oc in mk.get("outcomes", []):
                            point = oc.get("point")
                            if point is None or abs(float(point) - 2.5) > 1e-9:
                                continue
                            side = "over" if (oc.get("name") or "").lower() == "over" else "under"
                            price = oc.get("price")
                            if not price:
                                continue
                            slot = totals.setdefault(2.5, {})
                            if side not in slot or price > slot[side]:
                                slot[side] = float(price)
                                tot_books[(2.5, side)] = bk.get("title", "?")
            if not best:
                continue
            out.append({
                "home": ev.get("home_team"), "away": ev.get("away_team"),
                "commence": ev.get("commence_time"),
                "best_1x2": best, "books": books, "totals": totals,
                "n_books": len(ev.get("bookmakers", [])),
            })
        log.info("OddsAPI %s : %d matchs, %d books (appels cumulés: %d)",
                 league_code, len(out), max((o["n_books"] for o in out), default=0), self.calls)
        return out

    def meta(self) -> dict:
        return {"enabled": self.enabled, "calls": self.calls, "regions": self.regions}
