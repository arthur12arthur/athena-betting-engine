"""Cotes Pinnacle via l'API publique "guest" (gratuite, sans clé).

Pinnacle est la référence "sharp" du marché : ses cotes de fermeture sont
le meilleur estimateur public de la probabilité réelle. On les utilise
comme référence pour dé-viguer et détecter la valeur.

Endpoints (lecture seule, clé guest publique) :
  /0.1/sports/29/leagues                      -> ligues football
  /0.1/leagues/{id}/matchups                  -> matchs
  /0.1/matchups/{id}/markets/related/straight -> marchés (moneyline/total/spread)
"""
from __future__ import annotations

import re
from datetime import UTC, datetime

from ..config import League
from ..http import HttpClient
from ..logging_conf import get_logger
from .base import Fixture, OddsSnapshot

log = get_logger("source.pinnacle")

GUEST_KEY = "CmX2KcMrXuFmNg6YFbmTxE0y9CIrOi0R"
HOST = "https://guest.api.arcadia.pinnacle.com"
HEADERS = {
    "X-API-Key": GUEST_KEY,
    "Accept": "application/json",
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36",
    "Origin": "https://www.pinnacle.com",
    "Referer": "https://www.pinnacle.com/",
}


def american_to_decimal(price: float | str | None) -> float | None:
    """Cote américaine -> cote décimale (européenne)."""
    if price is None:
        return None
    try:
        p = float(price)
    except (TypeError, ValueError):
        return None
    if p == 0:
        return None
    dec = 1.0 + p / 100.0 if p > 0 else 1.0 + 100.0 / abs(p)
    return round(dec, 4) if dec > 1.0 else None


def _line_key(key: str) -> tuple[str, float | None]:
    """'s;0;ou;2.5' -> ('ou', 2.5) ; 's;0;m' -> ('ml', None)."""
    parts = (key or "").split(";")
    if len(parts) >= 3 and parts[2] == "m":
        return "ml", None
    if len(parts) >= 4 and parts[2] in {"ou", "sp"}:
        try:
            return ("ou" if parts[2] == "ou" else "sp"), float(parts[3])
        except ValueError:
            return parts[2], None
    return (parts[2] if len(parts) > 2 else ""), None


class PinnacleClient:
    def __init__(self, client: HttpClient | None = None):
        self.http = client or HttpClient()

    # ------------------------------------------------------------- endpoints
    def leagues(self) -> list[dict]:
        data = self.http.get(f"{HOST}/0.1/sports/29/leagues", headers=HEADERS, cache_hours=12)
        return data or []

    def matchups(self, league_id: int) -> list[dict]:
        data = self.http.get(f"{HOST}/0.1/leagues/{league_id}/matchups",
                             headers=HEADERS, cache_hours=0.5)
        if not data:
            return []
        return [m for m in data if m.get("parent") is None and m.get("type") == "matchup"]

    def markets(self, matchup_id: int) -> list[dict]:
        data = self.http.get(f"{HOST}/0.1/matchups/{matchup_id}/markets/related/straight",
                             headers=HEADERS, cache_hours=0.5)
        return data or []

    # ---------------------------------------------------------------- métier
    def fixtures_with_odds(self, lg: League, horizon_days: int = 4) -> list[Fixture]:
        if not lg.pinnacle:
            return []
        now = datetime.now(UTC)
        out: list[Fixture] = []
        for m in self.matchups(lg.pinnacle):
            try:
                start = datetime.fromisoformat(m["startTime"].replace("Z", "+00:00"))
            except Exception:  # noqa: BLE001
                continue
            delta_days = (start - now).total_seconds() / 86400
            if not (-1.0 <= delta_days <= horizon_days):
                continue
            participants = {p.get("alignment"): p.get("name") for p in m.get("participants", [])}
            home, away = participants.get("home"), participants.get("away")
            if not home or not away:
                continue
            fx = Fixture(
                league=lg.code,
                date_utc=start.isoformat(),
                home=home,
                away=away,
                source="pinnacle",
                external_id=str(m["id"]),
                status="live" if m.get("isLive") else "scheduled",
            )
            fx.extra["odds"] = self._odds_for(m["id"])
            out.append(fx)
        return out

    def _odds_for(self, matchup_id: int) -> OddsSnapshot:
        snap = OddsSnapshot(book="pinnacle", source="pinnacle", ts=datetime.now(UTC).isoformat())
        for mk in self.markets(matchup_id):
            # l'endpoint "related/straight" renvoie les marchés de TOUS les matchs
            # liés : on ne garde que ceux de CE match.
            if mk.get("matchupId") != matchup_id:
                continue
            if mk.get("period") not in (0, None) or mk.get("isAlternate"):
                continue
            if any(p.get("designation") is None for p in mk.get("prices", [])):
                continue
            kind, line = _line_key(mk.get("key", ""))
            prices = mk.get("prices", [])
            if kind == "ml":
                for p in prices:
                    des = (p.get("designation") or "").lower()
                    dec = american_to_decimal(p.get("price"))
                    if dec and des in {"home", "away", "draw"}:
                        snap.one_x_two[des] = dec
            elif kind == "ou" and line in {1.5, 2.5, 3.5}:
                for p in prices:
                    des = (p.get("designation") or "").lower()
                    dec = american_to_decimal(p.get("price"))
                    if dec and des in {"over", "under"}:
                        snap.totals.setdefault(line, {})[des] = dec
            elif kind == "sp" and line is not None:
                for p in prices:
                    des = (p.get("designation") or "").lower()
                    dec = american_to_decimal(p.get("price"))
                    if dec and des in {"home", "away"}:
                        snap.handicaps.setdefault(line, {})[des] = dec
        return snap


def discover_league_ids(league_names: list[str]) -> dict[str, int]:
    """Helper : retrouve les ids Pinnacle par nom (utilisé en maintenance)."""
    pc = PinnacleClient()
    idx = {lg.get("name", ""): lg.get("id") for lg in pc.leagues()}
    found: dict[str, int] = {}
    for want in league_names:
        if want in idx:
            found[want] = int(idx[want])
            continue
        pattern = re.compile(want, re.I)
        for name, lid in idx.items():
            if pattern.search(name):
                found[name] = int(lid)
                break
    return found
