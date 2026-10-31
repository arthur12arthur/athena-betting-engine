"""Calendrier & résultats via l'API publique ESPN (gratuite, sans clé).

Endpoint non documenté mais stable et largement utilisé. Sert à :
  - récupérer les matchs à venir (fenêtre de pronostic)
  - récupérer les scores pour le règlement des paris
"""
from __future__ import annotations

from datetime import date, timedelta

from ..config import Config
from ..http import HttpClient
from ..logging_conf import get_logger
from .base import Fixture

log = get_logger("source.espn")

SB = "https://site.api.espn.com/apis/site/v2/sports/soccer/{slug}/scoreboard"


def _parse(events: list[dict], league_code: str) -> list[Fixture]:
    out: list[Fixture] = []
    for ev in events or []:
        try:
            comp = (ev.get("competitions") or [{}])[0]
            status = (comp.get("status") or ev.get("status") or {}).get("type", {})
            home = away = None
            hg = ag = None
            for c in comp.get("competitors", []):
                name = (c.get("team") or {}).get("displayName") or c.get("team", {}).get("name")
                score = c.get("score")
                if c.get("homeAway") == "home":
                    home, hg = name, score
                else:
                    away, ag = name, score
            if not home or not away:
                continue
            st = "finished" if status.get("completed") else (
                "live" if status.get("state") == "in" else "scheduled"
            )
            if status.get("name") in {"STATUS_POSTPONED", "STATUS_CANCELED"}:
                st = "postponed"
            out.append(Fixture(
                league=league_code,
                date_utc=ev.get("date"),
                home=home,
                away=away,
                source="espn",
                external_id=str(ev.get("id")),
                status=st,
                home_goals=int(hg) if hg not in (None, "") else None,
                away_goals=int(ag) if ag not in (None, "") else None,
            ))
        except Exception as exc:  # noqa: BLE001
            log.debug("événement ESPN ignoré: %s", exc)
    return out


def fetch_scoreboard(cfg: Config, client: HttpClient, league_code: str,
                     day: date | None = None, span_days: int = 0) -> list[Fixture]:
    """Matchs d'une ligue sur [day, day+span_days]."""
    lg = cfg.league(league_code)
    if not lg.espn:
        return []
    d0 = day or date.today()
    url = SB.format(slug=lg.espn)
    # l'API soccer ESPN n'accepte qu'une seule date par requête -> on itère
    out: list[Fixture] = []
    for offset in range(span_days + 1):
        d = d0 + timedelta(days=offset)
        data = client.get(url, params={"dates": d.strftime("%Y%m%d"), "limit": 400}, cache_hours=1.5)
        if data:
            out.extend(_parse(data.get("events", []), league_code))
    return out


def fetch_upcoming(cfg: Config, client: HttpClient, horizon_days: int = 3) -> list[Fixture]:
    """Tous les matchs à venir sur l'horizon configuré, toutes ligues."""
    today = date.today()
    out: list[Fixture] = []
    for lg in cfg.leagues:
        if not lg.espn:
            continue
        got = fetch_scoreboard(cfg, client, lg.code, day=today, span_days=horizon_days)
        out.extend([f for f in got if f.status == "scheduled"])
    log.info("ESPN : %d matchs à venir (horizon %dj)", len(out), horizon_days)
    return out


def fetch_recent_results(cfg: Config, client: HttpClient, days_back: int = 3) -> list[Fixture]:
    """Résultats récents (règlement des paris)."""
    start = date.today() - timedelta(days=days_back)
    out: list[Fixture] = []
    for lg in cfg.leagues:
        if not lg.espn:
            continue
        out.extend(fetch_scoreboard(cfg, client, lg.code, day=start, span_days=days_back + 1))
    return [f for f in out if f.status == "finished"]
