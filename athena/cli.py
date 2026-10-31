"""Interface en ligne de commande ATHENA.

    athena doctor                  # vérifie configuration et sources
    athena ingest                  # télécharge et résume l'historique
    athena backtest                # backtest walk-forward -> site/data/backtest.json
    athena daily                   # pipeline complet (prédictions + paris + site)
    athena publish                 # régénère seulement le dashboard
    athena ratings                 # classements internes par ligue
    athena audit-aliases           # équipes non résolues entre sources
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from .config import load_config
from .logging_conf import setup_logging


def _cfg(args):
    return load_config(getattr(args, "config", None))


def cmd_doctor(args) -> int:
    cfg = _cfg(args)
    from .http import HttpClient
    from .sources.oddsapi import OddsApiClient
    print(f"ATHENA {cfg.raw.get('version')} — diagnostic")
    print(f"  ligues configurées : {len(cfg.leagues)}")
    print(f"  cache              : {cfg.cache_dir}")
    print(f"  bankroll initiale  : {cfg.betting.get('initial_bankroll')} u")
    print(f"  marchés            : {cfg.betting.get('markets')}")
    client = HttpClient(cache_dir=cfg.cache_dir)
    checks = []

    def check(name, fn):
        try:
            ok, detail = fn()
        except Exception as exc:  # noqa: BLE001
            ok, detail = False, str(exc)[:80]
        checks.append((name, ok, detail))
        print(f"  [{'OK ' if ok else 'KO '}] {name:<22} {detail}")

    def fd_check():
        from .sources import football_data_csv as fd
        df = fd.fetch_season("E0", datetime.utcnow().year if datetime.utcnow().month >= 7
                             else datetime.utcnow().year - 1, client)
        return (not df.empty), f"{len(df)} lignes"

    def espn_check():
        from datetime import date

        from .sources.espn import fetch_scoreboard
        fx = fetch_scoreboard(cfg, client, "E0", day=date.today())
        return True, f"{len(fx)} événements"

    def pinn_check():
        from .sources.pinnacle import PinnacleClient
        ms = PinnacleClient(client).matchups(1980)
        return len(ms) > 0, f"{len(ms)} matchs E0"

    def oddsapi_check():
        oa = OddsApiClient(client=client)
        return True, ("clé présente" if oa.enabled else "pas de clé (mode dégradé)")

    def gemini_check():
        from .sources.gemini import GeminiClient
        g = GeminiClient(cfg, client)
        return True, ("clé présente" if g.enabled else "pas de clé (texte désactivé)")

    check("football-data.co.uk", fd_check)
    check("ESPN", espn_check)
    check("Pinnacle", pinn_check)
    check("The Odds API", oddsapi_check)
    check("Gemini", gemini_check)
    bad = [c for c in checks if not c[1]]
    print(f"\n{len(checks) - len(bad)}/{len(checks)} sources opérationnelles")
    return 0 if len(bad) <= 1 else 1


def cmd_ingest(args) -> int:
    cfg = _cfg(args)
    from .http import HttpClient
    from .sources import football_data_csv as fd
    client = HttpClient(cache_dir=cfg.cache_dir)
    hist = fd.load_history(cfg, client, seasons_from=args.seasons_from)
    if hist.empty:
        print("aucune donnée récupérée")
        return 1
    out = cfg.root / "data" / "history_summary.json"
    summary = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_matches": int(len(hist)),
        "leagues": hist.groupby("league").size().to_dict(),
        "date_min": str(hist["date"].min().date()),
        "date_max": str(hist["date"].max().date()),
        "odds_avg_coverage": round(float(hist["odds_avg_home"].notna().mean()), 4),
        "odds_psc_coverage": round(float(hist["odds_psc_home"].notna().mean()), 4),
        "xg_coverage": round(float(hist["xg_home"].notna().mean()), 4),
    }
    out.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


def cmd_backtest(args) -> int:
    cfg = _cfg(args)
    from .backtest.engine import run
    from .etl.features import prepare
    from .http import HttpClient
    from .report.dashboard import render
    from .sources import football_data_csv as fd
    client = HttpClient(cache_dir=cfg.cache_dir)
    hist = fd.load_history(cfg, client, seasons_from=args.seasons_from)
    feats, _ = prepare(hist, None, {lg.code: i for i, lg in enumerate(cfg.leagues)})
    seasons = [int(s) for s in args.seasons.split(",")] if args.seasons else None
    report = run(cfg, hist, feats, test_seasons=seasons)
    out = Path(args.out) if args.out else (cfg.site_dir / "data" / "backtest.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=float, ensure_ascii=False), encoding="utf-8")
    p, pr = report.get("performance", {}), report.get("predictions", {})
    print("\n=== RÉSULTATS ===")
    print(f"prédictions : n={pr.get('n')} logloss={pr.get('logloss')} brier={pr.get('brier')} "
          f"rps={pr.get('rps')} acc={pr.get('accuracy')}")
    print(f"paris       : n={p.get('n_bets')} ROI={p.get('roi')} P&L={p.get('pnl')} "
          f"drawdown={p.get('max_drawdown')} CLV={p.get('clv_avg')}")
    print(f"-> {out}")
    if args.render:
        render(cfg)
    return 0


def cmd_daily(args) -> int:
    cfg = _cfg(args)
    from .live import run_daily
    from .report.dashboard import render
    out = run_daily(cfg, do_publish=not args.no_publish,
                    use_llm=False if args.no_llm else None)
    if not args.no_publish:
        render(cfg)
    print(json.dumps(out, indent=1, default=float))
    return 0


def cmd_publish(args) -> int:
    cfg = _cfg(args)
    from .report.dashboard import render
    p = render(cfg)
    print(f"dashboard -> {p}")
    return 0


def cmd_ratings(args) -> int:
    cfg = _cfg(args)
    from .http import HttpClient
    from .models.dixon_coles import fit_leagues
    from .sources import football_data_csv as fd
    client = HttpClient(cache_dir=cfg.cache_dir)
    hist = fd.load_history(cfg, client, seasons_from=args.seasons_from)
    models = fit_leagues(hist, xi=float(cfg.model.get("xi_decay_per_day", 0.0019)),
                         ridge=float(cfg.model.get("ridge", 0.035)),
                         max_goals=int(cfg.model.get("max_goals", 10)))
    for code, m in models.items():
        print(f"\n{cfg.league(code).label} — γ={m.home_adv:.2f}, ρ={m.rho:+.3f}, {m.n_matches} matchs")
        print(m.team_strength().head(args.top).to_string(index=False))
    return 0


def cmd_audit_aliases(args) -> int:
    cfg = _cfg(args)
    from collections import defaultdict

    from .etl.names import TeamResolver, similarity
    from .http import HttpClient
    from .sources import football_data_csv as fd
    from .sources.espn import fetch_upcoming
    from .sources.pinnacle import PinnacleClient
    client = HttpClient(cache_dir=cfg.cache_dir)
    hist = fd.load_history(cfg, client)
    teams = defaultdict(set)
    for r in hist.itertuples():
        teams[r.league].add(r.home)
        teams[r.league].add(r.away)
    teams = {k: sorted(v) for k, v in teams.items()}
    resolver = TeamResolver.from_config(cfg, teams_by_league=teams)
    names: list[tuple[str, str]] = [(f.league, f.home) for f in fetch_upcoming(cfg, client, 5)]
    names += [(f.league, f.away) for f in fetch_upcoming(cfg, client, 5)]
    pc = PinnacleClient(client)
    for lg in cfg.leagues:
        for f in pc.fixtures_with_odds(lg, horizon_days=5):
            names += [(f.league, f.home), (f.league, f.away)]
    unresolved = []
    for league, name in names:
        if resolver.resolve(name, league) is None:
            best = max(teams.get(league, ["?"]), key=lambda t: similarity(name, t))
            unresolved.append({"league": league, "raw": name, "suggest": best,
                               "score": round(similarity(name, best), 2)})
    if not unresolved:
        print("Tous les noms sont résolus.")
        return 0
    seen = set()
    for u in unresolved:
        key = (u["league"], u["raw"])
        if key in seen:
            continue
        seen.add(key)
        print(f'  {u["league"]:4s} "{u["raw"]}" -> suggérer "{u["suggest"]}" ({u["score"]})')
    print(f"\n{len(seen)} noms à ajouter dans data/aliases.yaml")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="athena", description="ATHENA — moteur de pronostics football")
    p.add_argument("--config", help="chemin d'un config.yaml alternatif")
    p.add_argument("--log-level", default=None)
    sub = p.add_subparsers(dest="command", required=True)

    for name, fn, help_ in [
        ("doctor", cmd_doctor, "diagnostic des sources et de la configuration"),
        ("ingest", cmd_ingest, "télécharge l'historique et résume"),
        ("backtest", cmd_backtest, "backtest walk-forward"),
        ("daily", cmd_daily, "pipeline quotidien complet"),
        ("publish", cmd_publish, "régénère le dashboard"),
        ("ratings", cmd_ratings, "classements internes"),
        ("audit-aliases", cmd_audit_aliases, "noms d'équipes non résolus"),
    ]:
        sp = sub.add_parser(name, help=help_)
        sp.set_defaults(func=fn)
        if name in {"ingest", "backtest", "ratings"}:
            sp.add_argument("--seasons-from", type=int, default=None)
        if name == "backtest":
            sp.add_argument("--seasons", help="ex. 2024,2025,2026")
            sp.add_argument("--out", help="chemin du rapport JSON")
            sp.add_argument("--render", action="store_true", help="régénère aussi le dashboard")
        if name == "daily":
            sp.add_argument("--no-publish", action="store_true")
            sp.add_argument("--no-llm", action="store_true")
        if name == "ratings":
            sp.add_argument("--top", type=int, default=8)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
