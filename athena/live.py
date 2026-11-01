"""Pipeline quotidien ATHENA : ingestion -> règlement -> prédiction -> value -> publication.

Séquence exécutée chaque jour par GitHub Actions (et à la demande en CLI) :

  1. ``ingest``    : résultats/cotes récents (football-data.co.uk)
  2. ``settle``    : règle les paris ouverts (P&L exact, demi-lignes incluses)
  3. ``fixtures``  : calendrier (ESPN) + cotes (Pinnacle, OddsAPI optionnel)
  4. ``predict``   : matrices de scores -> 1X2, over/under, handicap
  5. ``select``    : value betting multi-marchés + Kelly fractionnaire
  6. ``archive``   : historique des pronostics (résultats vérifiables)
  7. ``publish``   : JSON publics, dashboard, résumé de run, Telegram

Idempotent : relancer deux fois le même jour ne duplique ni pari ni pronostic.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime

import pandas as pd

from .betting.devig import blend_with_market
from .betting.offers import Offer, offers_from_live
from .betting.settlement import settle_pnl
from .betting.staking import plan_stake
from .betting.value import label
from .config import Config
from .etl.features import prepare
from .etl.names import TeamResolver
from .history import PredictionArchive
from .http import HttpClient
from .logging_conf import get_logger
from .models.ensemble import train
from .models.markets import parse_market
from .sources import football_data_csv as fd
from .sources.espn import fetch_recent_results, fetch_upcoming
from .sources.gemini import GeminiClient
from .sources.oddsapi import OddsApiClient
from .sources.pinnacle import PinnacleClient
from .sources.telegram import TelegramNotifier
from .state import Ledger

log = get_logger("live")

STATUS_ICON = {"won": "✅", "lost": "❌", "push": "↩️", "half_won": "🟢", "half_lost": "🟠"}


# ==========================================================================
# Utilitaires
# ==========================================================================
def bet_id(league: str, kickoff: str, home: str, away: str, market: str, selection: str) -> str:
    raw = f"{league}|{kickoff[:10]}|{home}|{away}|{market}|{selection}"
    return hashlib.sha1(raw.encode()).hexdigest()[:14]


def results_map(history: pd.DataFrame, espn_results: list | None = None) -> dict[tuple, tuple[int, int]]:
    """{(ligue, date, domicile, extérieur): (buts_dom, buts_ext)}"""
    out: dict[tuple, tuple[int, int]] = {}
    for r in history.dropna(subset=["home_goals", "away_goals"]).itertuples():
        out[(str(r.league), str(pd.Timestamp(r.date).date()), r.home, r.away)] = (
            int(r.home_goals), int(r.away_goals))
    for f in (espn_results or []):
        if f.status != "finished" or f.home_goals is None:
            continue
        out.setdefault((f.league, str(pd.Timestamp(f.date_utc).date()), f.home, f.away),
                       (int(f.home_goals), int(f.away_goals)))
    return out


def _find_score(results: dict, league: str, kickoff: str, home: str, away: str):
    for delta in (0, 1, -1, 2, -2):
        day = str((pd.Timestamp(kickoff) + pd.Timedelta(days=delta)).date())
        if (league, day, home, away) in results:
            return results[(league, day, home, away)]
    return None


# ==========================================================================
# 1. Ingestion
# ==========================================================================
def ingest(cfg: Config, client: HttpClient) -> pd.DataFrame:
    history = fd.load_history(cfg, client)
    if history.empty:
        raise RuntimeError("historique indisponible (réseau ?)")
    return history


# ==========================================================================
# 2. Règlement
# ==========================================================================
def settle_open_bets(cfg: Config, ledger: Ledger, results: dict) -> list[dict]:
    """Règle les paris ouverts. Renvoie la liste des paris réglés (pour le rapport)."""
    settled: list[dict] = []
    for b in ledger.open_bets():
        score = _find_score(results, b["league"], b["kickoff"], b["home"], b["away"])
        if score is None:
            continue
        hg, ag = score
        pnl, status = settle_pnl(b["market"], b["selection"], b.get("line"),
                                 float(b["stake"]), float(b["price"]), hg, ag)
        ledger.settle_bet(b["bet_id"], status, pnl, result=f"{hg}-{ag}",
                          closing_price=b.get("closing_price"))
        b = next(x for x in ledger.bets if x["bet_id"] == b["bet_id"])
        settled.append(b)
        log.info("réglé : %s–%s %s (%s) → %s %+.2f", b["home"], b["away"], b["label"],
                 f"{hg}-{ag}", status, pnl)
    if settled:
        ledger.save()
    return settled


# ==========================================================================
# 3. Fixtures + cotes
# ==========================================================================
@dataclass
class FixtureBoard:
    rows: list[dict] = field(default_factory=list)


def build_fixture_board(cfg: Config, client: HttpClient, resolver: TeamResolver,
                        horizon_days: int) -> FixtureBoard:
    """Fusionne calendrier (ESPN), cotes 1X2/totals/handicaps (Pinnacle) et multi-books."""
    board: dict[tuple, dict] = {}

    def slot(league: str, date: str, home: str, away: str) -> dict:
        key = (league, home, away)
        if key not in board:
            board[key] = {"league": league, "date": date, "home": home, "away": away,
                          "sources": [], "pinnacle_odds": {}, "pinnacle_totals": {},
                          "pinnacle_handicaps": {}, "best_odds": {}, "best_totals": {},
                          "best_handicaps": {}, "books": {}, "n_books": 0}
        return board[key]

    for f in fetch_upcoming(cfg, client, horizon_days):
        home = resolver.resolve(f.home, f.league)
        away = resolver.resolve(f.away, f.league)
        if not home or not away:
            continue
        s = slot(f.league, str(pd.Timestamp(f.date_utc).date()), home, away)
        s["sources"].append("espn")
        s["kickoff_utc"] = f.date_utc

    pc = PinnacleClient(client)
    for lg in cfg.leagues:
        for f in pc.fixtures_with_odds(lg, horizon_days=horizon_days):
            home = resolver.resolve(f.home, lg.code)
            away = resolver.resolve(f.away, lg.code)
            if not home or not away:
                continue
            snap = f.extra.get("odds")
            s = slot(lg.code, str(pd.Timestamp(f.date_utc).date()), home, away)
            s["sources"].append("pinnacle")
            s.setdefault("kickoff_utc", f.date_utc)
            if snap and snap.is_usable():
                s["pinnacle_odds"] = dict(snap.one_x_two)
                s["pinnacle_totals"] = {f"{k:g}": v for k, v in snap.totals.items()}
                s["pinnacle_handicaps"] = {f"{k:g}": v for k, v in snap.handicaps.items()}
                s["overround"] = round(snap.overround, 4)

    oa = OddsApiClient(client=client)
    if oa.enabled:
        for lg in cfg.leagues:
            for ev in oa.odds_for_league(lg.code):
                home = resolver.resolve(ev["home"], lg.code)
                away = resolver.resolve(ev["away"], lg.code)
                if not home or not away:
                    continue
                date = (str(pd.Timestamp(ev["commence"]).tz_convert("UTC").date())
                        if ev.get("commence") else "")
                s = slot(lg.code, date, home, away)
                s["sources"].append("oddsapi")
                s["best_odds"] = ev.get("best_1x2", {})
                s["books"] = ev.get("books", {})
                s["n_books"] = ev.get("n_books", 0)
                if ev.get("totals", {}).get(2.5):
                    s["best_totals"] = {"2.5": ev["totals"][2.5]}

    out = FixtureBoard(rows=[v for v in board.values() if v.get("kickoff_utc")])
    out.rows.sort(key=lambda r: (r["date"], r["league"]))
    log.info("board : %d matchs (%d avec cotes Pinnacle, %d avec totaux, %d avec handicaps)",
             len(out.rows), sum(1 for r in out.rows if r["pinnacle_odds"]),
             sum(1 for r in out.rows if r["pinnacle_totals"]),
             sum(1 for r in out.rows if r["pinnacle_handicaps"]))
    return out


# ==========================================================================
# 4. Prédiction
# ==========================================================================
@dataclass
class ModelContext:
    ensemble: object
    resolver: TeamResolver
    history: pd.DataFrame
    features: pd.DataFrame


def build_model(cfg: Config, history: pd.DataFrame) -> ModelContext:
    league_map = {lg.code: i for i, lg in enumerate(cfg.leagues)}
    feats, _ = prepare(history, None, league_map)
    cutoff = pd.Timestamp(history["date"].max()) - pd.Timedelta(days=60)
    tr = feats[feats["date"] <= cutoff]
    va = feats[feats["date"] > cutoff]
    if len(tr) < 3000:
        tr, va = feats, None
    ens = train(cfg, history[history["date"] <= (cutoff if va is not None else history["date"].max())],
                tr, va)
    teams: dict[str, list[str]] = {}
    for league, grp in history.groupby("league"):
        teams[str(league)] = sorted(set(grp["home"]) | set(grp["away"]))
    resolver = TeamResolver.from_config(cfg, teams_by_league=teams)
    return ModelContext(ensemble=ens, resolver=resolver, history=history, features=feats)


def predict_board(cfg: Config, ctx: ModelContext, board: FixtureBoard) -> pd.DataFrame:
    rows = board.rows
    if not rows:
        return pd.DataFrame()
    fixtures = pd.DataFrame([{"league": r["league"], "date": r["kickoff_utc"],
                              "home": r["home"], "away": r["away"]} for r in rows])
    fixtures["date"] = pd.to_datetime(fixtures["date"], utc=True).dt.tz_localize(None)
    league_map = {lg.code: i for i, lg in enumerate(cfg.leagues)}
    _, tgt = prepare(ctx.history, fixtures, league_map)
    if tgt.empty:
        return pd.DataFrame()
    mats = ctx.ensemble.predict_frame(tgt)
    out = []
    for (_, feat), mat in zip(tgt.iterrows(), mats, strict=False):
        p = mat.probs_1x2()
        ou = mat.market("OU2.5")
        out.append({
            "league": feat["league"], "home": feat["home"], "away": feat["away"],
            "kickoff": str(pd.Timestamp(feat["date"]).date()),
            "p_home": p["home"], "p_draw": p["draw"], "p_away": p["away"],
            "p_ou25_over": ou["over"], "p_ou25_under": ou["under"],
            "p_btts_yes": mat.market("BTTS")["yes"],
            "p_dc_1x": mat.market("DC")["1X"],
            "lam_home": mat.lam_home, "lam_away": mat.lam_away,
            "top_scores": mat.top_scores(3), "matrix": mat,
            "elo_home": float(feat.get("elo_home", 0)), "elo_away": float(feat.get("elo_away", 0)),
            "form5_home": float(feat.get("form5_home", 0)), "form5_away": float(feat.get("form5_away", 0)),
            "xg5_home": float(feat.get("xg5_for_home", 0)), "xg5_away": float(feat.get("xg5_for_away", 0)),
        })
    return pd.DataFrame(out)


# ==========================================================================
# 5. Sélection des paris (multi-marchés)
# ==========================================================================
def _market_params(bcfg: dict, market: str) -> dict:
    """Fusionne les paramètres globaux et les surcharges par marché."""
    family, _ = parse_market(market)
    overrides = (bcfg.get("market_params") or {}).get(market) or \
                (bcfg.get("market_params") or {}).get(family) or {}
    return {**bcfg, **overrides}


def select_bets(cfg: Config, preds: pd.DataFrame, board: FixtureBoard,
                ledger: Ledger) -> list[dict]:
    if preds.empty:
        return []
    bcfg = cfg.betting
    lam = float(cfg.model.get("shrink_to_market", 0.70))
    markets = list(bcfg.get("markets", ["1X2"]))
    board_idx = {(r["league"], r["home"], r["away"]): r for r in board.rows}
    min_books = int(bcfg.get("min_books_multi", 3))

    bankroll = ledger.bankroll
    today = str(datetime.now(UTC).date())
    day_bets = [b for b in ledger.bets if str(b.get("created_at", ""))[:10] == today]
    day_exposure = sum(float(b["stake"]) for b in day_bets)
    max_exp = float(bcfg.get("max_daily_exposure_pct", 0.07)) * bankroll
    ss = bcfg.get("single_source", {}) or {}

    candidates: list[dict] = []
    for _, pr in preds.iterrows():
        row = board_idx.get((pr["league"], pr["home"], pr["away"]))
        if not row:
            continue
        offers = offers_from_live(row, markets, min_books)
        if not offers:
            continue
        multi_book = bool(row.get("best_odds")) and row.get("n_books", 0) >= min_books
        if not multi_book and not ss.get("enabled", True):
            continue
        stake_scale = 1.0 if multi_book else float(ss.get("stake_scale", 0.5))
        default_min_edge = float(bcfg.get("min_edge", 0.03)) if multi_book \
            else float(ss.get("min_edge", 0.05))

        # regroupement par (marché, ligne) pour blanchir les probabilités sharp
        pools: dict[tuple[str, float | None], list[Offer]] = {}
        for o in offers:
            pools.setdefault((o.market, o.line), []).append(o)

        for (market, line), group in pools.items():
            params = _market_params(bcfg, market)
            family, ln = parse_market(market, line)
            try:
                p_model = pr["matrix"].market(family, ln)
            except Exception:  # noqa: BLE001
                continue
            p_sharp = {o.selection: o.p_sharp for o in group if o.p_sharp}
            if not p_sharp:
                continue
            p_fair = blend_with_market(p_model, p_sharp, lam)
            min_edge = float(params.get("min_edge", default_min_edge))
            for o in group:
                p = p_fair.get(o.selection)
                if p is None:
                    continue
                price = float(o.price_bet)
                edge = p * price - 1.0
                if edge < min_edge or edge > float(params.get("max_edge", 0.10)):
                    continue
                ratio = p / o.p_sharp if o.p_sharp else 1.0
                if not (float(bcfg.get("max_divergence_low", 0.75)) <= ratio
                        <= float(bcfg.get("max_divergence_high", 1.35))):
                    continue
                if not (float(params.get("min_odds", 1.45)) <= price
                        <= float(params.get("max_odds", 8.0))):
                    continue
                if not (float(params.get("min_prob", 0.06)) <= p <= float(params.get("max_prob", 0.80))):
                    continue
                candidates.append({
                    "league": pr["league"], "home": pr["home"], "away": pr["away"],
                    "kickoff": pr["kickoff"], "market": market, "selection": o.selection,
                    "line": ln, "label": label(family, o.selection, ln),
                    "price": round(price, 3), "p_model": round(float(p_model.get(o.selection, 0.0)), 4),
                    "p_fair": round(float(p), 4), "p_sharp": round(float(o.p_sharp or 0.0), 4),
                    "edge": round(edge, 4), "book": o.book or "pinnacle",
                    "n_books": row.get("n_books", 1), "multi_book": multi_book,
                    "stake_scale": stake_scale,
                })

    candidates.sort(key=lambda c: -c["edge"])
    chosen: list[dict] = []
    seen_matches: dict[str, int] = {}
    max_per_match = int(bcfg.get("max_bets_per_match", 1))
    for c in candidates:
        if len(chosen) >= int(bcfg.get("max_bets_per_day", 8)):
            break
        mkey = f'{c["league"]}|{c["home"]}|{c["away"]}'
        if seen_matches.get(mkey, 0) >= max_per_match:
            continue
        if ledger.has_bet(bet_id(c["league"], c["kickoff"], c["home"], c["away"],
                                 c["market"], c["selection"])):
            continue
        remaining = max(0.0, max_exp - day_exposure)
        plan = plan_stake(c["p_fair"], c["price"], bankroll, bcfg, remaining_exposure=remaining)
        if plan.stake <= 0:
            continue
        scaled = round(plan.stake * float(c["stake_scale"]), 2)
        if scaled <= 0:
            continue
        c.update({"stake": scaled, "kelly_full": plan.kelly_full,
                  "bet_id": bet_id(c["league"], c["kickoff"], c["home"], c["away"],
                                   c["market"], c["selection"]),
                  "created_at": datetime.now(UTC).isoformat(timespec="seconds")})
        day_exposure += scaled
        seen_matches[mkey] = seen_matches.get(mkey, 0) + 1
        chosen.append(c)
        if day_exposure >= max_exp:
            break
    log.info("value : %d paris retenus (sur %d candidats, marchés %s)",
             len(chosen), len(candidates), ",".join(markets))
    return chosen


# ==========================================================================
# 6. Publication
# ==========================================================================
def publish(cfg: Config, preds: pd.DataFrame, board: FixtureBoard, ledger: Ledger,
            new_bets: list[dict], model_info: dict, archive: PredictionArchive,
            briefings: dict[str, str] | None = None) -> dict:
    site = cfg.site_dir
    data_dir = site / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    briefings = briefings or {}

    payload = []
    for _, pr in preds.iterrows():
        entry = {
            "league": pr["league"], "home": pr["home"], "away": pr["away"],
            "kickoff": pr["kickoff"],
            "p": {"home": round(pr["p_home"], 4), "draw": round(pr["p_draw"], 4),
                  "away": round(pr["p_away"], 4)},
            "xg": {"home": round(pr["lam_home"], 2), "away": round(pr["lam_away"], 2)},
            "fair": {k: round(1 / max(pr[f"p_{k}"], 1e-6), 2) for k in ("home", "draw", "away")},
            "ou25": {"over": round(pr["p_ou25_over"], 4), "under": round(pr["p_ou25_under"], 4),
                     "fair_over": round(1 / max(pr["p_ou25_over"], 1e-6), 2),
                     "fair_under": round(1 / max(pr["p_ou25_under"], 1e-6), 2)},
            "btts": round(pr["p_btts_yes"], 4),
            "scores": [[s, round(p, 4)] for s, p in pr["top_scores"]],
            "elo": [round(pr["elo_home"]), round(pr["elo_away"])],
            "form": [round(pr["form5_home"], 2), round(pr["form5_away"], 2)],
        }
        row = next((r for r in board.rows if r["league"] == pr["league"]
                    and r["home"] == pr["home"] and r["away"] == pr["away"]), None)
        if row:
            if row.get("pinnacle_odds"):
                entry["sharp_odds"] = {k: round(v, 2) for k, v in row["pinnacle_odds"].items()}
            if row.get("pinnacle_totals", {}).get("2.5"):
                entry["sharp_totals"] = {k: round(v, 2)
                                         for k, v in row["pinnacle_totals"]["2.5"].items()}
            if row.get("pinnacle_handicaps"):
                best_line = min((float(k) for k in row["pinnacle_handicaps"]),
                                key=lambda x: (abs(x), abs(x - round(x))), default=None)
                if best_line is not None:
                    entry["sharp_handicap"] = {
                        "line": best_line,
                        **{k: round(v, 2)
                           for k, v in row["pinnacle_handicaps"][f"{best_line:g}"].items()},
                    }
            if row.get("best_odds"):
                entry["best_odds"] = {k: round(v, 2) for k, v in row["best_odds"].items()}
                entry["n_books"] = row.get("n_books", 0)
        # handicap asiatique : on publie la ligne la plus proche du marché sharp
        ah_line = float((entry.get("sharp_handicap") or {}).get("line", 0.0))
        try:
            ahm = pr["matrix"].market("AH", ah_line)
            entry["ah"] = {"line": ah_line,
                           "home": round(float(ahm["home"]), 4),
                           "away": round(float(ahm["away"]), 4),
                           "fair_home": round(1 / max(float(ahm["home"]), 1e-6), 2),
                           "fair_away": round(1 / max(float(ahm["away"]), 1e-6), 2)}
        except Exception:  # noqa: BLE001
            entry["ah"] = None
        key = f'{pr["league"]}:{pr["home"]}-{pr["away"]}'
        if key in briefings:
            entry["briefing"] = briefings[key]
        payload.append(entry)

    (data_dir / "predictions.json").write_text(json.dumps({
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": model_info, "fixtures": payload, "bets": new_bets,
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    (data_dir / "ledger.json").write_text(json.dumps({
        "stats": ledger.stats(), "bets": ledger.bets[-1000:],
        "equity": ledger.data.get("equity", [])[-600:],
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    (data_dir / "history.json").write_text(json.dumps({
        "stats": archive.stats(),
        "stats_30d": archive.stats(last_days=30),
        "recent": archive.recent(120),
        "by_league": archive.stats().get("by_league", []),
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    return {"n_fixtures": len(payload), "n_bets": len(new_bets)}


def write_run_summary(cfg: Config, *, started: datetime, fixtures: int, new_bets: list[dict],
                      settled: list[dict], ledger: Ledger, archive: PredictionArchive,
                      model_info: dict, telegram_sent: bool) -> str:
    """Résumé Markdown affiché dans GitHub Actions (Step Summary) et en console."""
    st = ledger.stats()
    lines = [
        f"## 🏛️ ATHENA — exécution du {datetime.now(UTC):%d/%m/%Y %H:%M} UTC",
        "",
        f"**Bankroll papier : {st['bankroll']:.2f} u** ({st['pnl_total']:+.2f} u · "
        f"ROI {st['roi']*100:+.2f} %) · "
        f"{st['n_settled']} paris réglés · {st['n_open']} en cours · "
        f"modèle `{model_info['version']}`",
        "",
        "| Indicateur | Valeur |",
        "|---|---|",
        f"| Matchs analysés | {fixtures} |",
        f"| Value bets placés | {len(new_bets)} |",
        f"| Paris réglés ce run | {len(settled)} |",
        f"| Exposition ouverte | {st['exposure']:.2f} u |",
        f"| CLV moyen (cumulé) | {(st['clv_avg']*100):+.2f} % |" if st["clv_avg"] is not None
        else "| CLV moyen (cumulé) | n/a |",
        f"| Telegram | {'envoyé' if telegram_sent else 'non configuré / désactivé'} |",
        "",
    ]
    if settled:
        lines += ["### ✅ Résultats réglés", "",
                  "| Match | Score | Pari | Cote | Mise | P&L | Statut |",
                  "|---|---|---|---|---|---|---|"]
        for b in settled[:20]:
            icon = STATUS_ICON.get(b["status"], "•")
            lines.append(
                f"| {b['home']} – {b['away']} | **{b.get('result', '')}** | {b['label']} | "
                f"{b['price']:.2f} | {b['stake']:.2f} u | {b['pnl']:+.2f} u | {icon} {b['status']} |")
        lines.append("")
    if new_bets:
        lines += ["### 🎯 Value bets du jour", "",
                  "| Match | Coup d'envoi | Marché | Cote | Edge | Mise | Source |",
                  "|---|---|---|---|---|---|---|"]
        for b in new_bets[:20]:
            src = "multi-books" if b.get("multi_book") else "sharp seul"
            lines.append(
                f"| {b['home']} – {b['away']} | {b['kickoff']} | {b['label']} | "
                f"**{b['price']:.2f}** | {b['edge']*100:+.1f} % | {b['stake']:.2f} u | {src} |")
        lines.append("")
    pstats = archive.stats()
    if pstats["n"]:
        lines += ["### 📈 Historique des pronostics", "",
                  f"- Pronostics résolus : **{pstats['n']}** · réussite "
                  f"**{(pstats['hit_rate'] or 0)*100:.1f} %** · "
                  f"log loss **{pstats['logloss']:.4f}** · Brier **{pstats['brier']:.4f}**"]
        for row in pstats["by_league"][:6]:
            lines.append(f"  - {row['league']} : {row['n']} matchs, "
                         f"{row['hit_rate']*100:.1f} % de bons pronostics")
        lines.append("")
    lines += [f"_durée {(datetime.now(UTC)-started).total_seconds():.0f} s · "
              f"aucune mise réelle · 18+_"]
    text = "\n".join(lines)
    (cfg.site_dir / "data").mkdir(parents=True, exist_ok=True)
    (cfg.site_dir / "data" / "run_summary.md").write_text(text, encoding="utf-8")
    return text


# ==========================================================================
# Orchestration
# ==========================================================================
def run_daily(cfg: Config, do_publish: bool = True, use_llm: bool | None = None,
              notify: bool = True) -> dict:
    started = datetime.now(UTC)
    client = HttpClient(cache_dir=cfg.cache_dir)
    ledger = Ledger.load(cfg.root / "data" / "state" / "ledger.json",
                         float(cfg.betting.get("initial_bankroll", 1000.0)))
    archive = PredictionArchive.load(cfg.root / "data" / "state" / "predictions_history.json")

    log.info("=== ATHENA : exécution quotidienne ===")
    history = ingest(cfg, client)
    espn_results = fetch_recent_results(cfg, client, 4)
    res = results_map(history, espn_results)

    settled = settle_open_bets(cfg, ledger, res)
    n_arch = archive.settle(res)
    if n_arch:
        log.info("historique : %d pronostics résolus", n_arch)

    ctx = build_model(cfg, history)
    board = build_fixture_board(cfg, client, ctx.resolver, int(cfg.publish.get("horizon_days", 3)))
    preds = predict_board(cfg, ctx, board)

    # CLV : mémorise la dernière cote sharp vue avant le coup d'envoi
    updated = 0
    for b in ledger.open_bets():
        for row in board.rows:
            if (row["league"], row["home"], row["away"]) == (b["league"], b["home"], b["away"]):
                if b["market"] == "1X2":
                    price = (row.get("pinnacle_odds") or {}).get(b["selection"])
                elif b["market"].startswith("OU") or b["market"] == "OU25":
                    tot = (row.get("pinnacle_totals") or {}).get(
                        f"{float(b.get('line') or 2.5):g}") or {}
                    price = tot.get(b["selection"])
                else:
                    hc = (row.get("pinnacle_handicaps") or {}).get(
                        f"{float(b.get('line') or 0.0):g}") or {}
                    price = hc.get(b["selection"])
                if price:
                    b["closing_price"] = round(float(price), 3)
                    updated += 1
                break
    if updated:
        ledger.save()

    new_bets = select_bets(cfg, preds, board, ledger)
    for b in new_bets:
        ledger.add_bet(b)
    if new_bets:
        ledger.save()

    if not preds.empty:
        archive.add_many([{"league": p["league"], "kickoff": p["kickoff"], "home": p["home"],
                           "away": p["away"], "p_home": p["p_home"], "p_draw": p["p_draw"],
                           "p_away": p["p_away"], "lam_home": p["lam_home"],
                           "lam_away": p["lam_away"]} for _, p in preds.iterrows()])

    # briefings LLM (optionnels, uniquement sur les paris)
    briefings: dict[str, str] = {}
    use_llm = cfg.llm.get("enabled", True) if use_llm is None else use_llm
    if use_llm and new_bets:
        gem = GeminiClient(cfg, client)
        if gem.enabled:
            for b in new_bets[:6]:
                pr = preds[(preds["league"] == b["league"]) & (preds["home"] == b["home"])
                           & (preds["away"] == b["away"])]
                if pr.empty:
                    continue
                pr = pr.iloc[0]
                txt = gem.match_briefing(
                    cfg.league(b["league"]).label, b["home"], b["away"], b["kickoff"],
                    {"home": pr["p_home"], "draw": pr["p_draw"], "away": pr["p_away"]},
                    {"home": pr["lam_home"], "away": pr["lam_away"]},
                    value_note=f"Value détectée : {b['label']} à {b['price']} "
                               f"(edge {b['edge']:+.1%}).")
                if txt:
                    briefings[f'{b["league"]}:{b["home"]}-{b["away"]}'] = txt

    ens = ctx.ensemble
    model_info = {
        "version": ens.version, "trained_at": ens.trained_at,
        "w_dc": round(ens.w_dc, 3), "temperature": round(ens.temperature, 3),
        "calibration": ens.metrics,
        "gbm_val": ens.gbm.metrics if ens.gbm else {},
        "n_matches_train": int(len(history)), "n_leagues": len(ens.dc),
        "markets": list(cfg.betting.get("markets", ["1X2"])),
        "leagues": [lg.code for lg in cfg.leagues],
    }

    if do_publish:
        publish(cfg, preds, board, ledger, new_bets, model_info, archive, briefings)
        ratings = top_ratings(ctx)
        (cfg.site_dir / "data" / "ratings.json").write_text(
            json.dumps(ratings, ensure_ascii=False, indent=1), encoding="utf-8")
    archive.save()

    # ---- rapport de run + Telegram
    summary_md = write_run_summary(cfg, started=started, fixtures=len(preds),
                                   new_bets=new_bets, settled=settled, ledger=ledger,
                                   archive=archive, model_info=model_info,
                                   telegram_sent=False)
    tg_sent = False
    if notify:
        tg = TelegramNotifier.from_config(cfg)
        if tg.enabled:
            top = _top_picks(preds)
            msg = tg.daily_message(fixtures=len(preds), bets=new_bets, settled=settled,
                                   stats=ledger.stats(), model=model_info, top=top,
                                   n_pred_archive=archive.stats().get("n", 0),
                                   hit_rate=archive.stats().get("hit_rate"))
            tg_sent = tg.send(msg)
            if tg_sent:  # régénère le résumé avec la mention "envoyé"
                write_run_summary(cfg, started=started, fixtures=len(preds), new_bets=new_bets,
                                  settled=settled, ledger=ledger, archive=archive,
                                  model_info=model_info, telegram_sent=True)

    out = {"fixtures": len(preds), "bets": len(new_bets), "settled": len(settled),
           "archived": len(archive.entries), "bankroll": ledger.bankroll,
           "model": model_info["version"], "telegram": tg_sent}
    ledger.log_run({"fixtures": len(preds), "bets": len(new_bets), "settled": len(settled),
                    "telegram": tg_sent,
                    "duration_s": round((datetime.now(UTC) - started).total_seconds(), 1)})
    ledger.save()
    print("\n" + summary_md + "\n")
    log.info("=== terminé en %.1fs : %d matchs, %d paris, %d réglés, bankroll %.2f ===",
             (datetime.now(UTC) - started).total_seconds(),
             len(preds), len(new_bets), len(settled), ledger.bankroll)
    return out


def _top_picks(preds: pd.DataFrame, n: int = 5) -> list[dict]:
    """Les matchs où le modèle est le plus tranché (pour le message Telegram)."""
    if preds.empty:
        return []
    ordered = preds.assign(_m=preds[["p_home", "p_draw", "p_away"]].max(axis=1)) \
        .sort_values("_m", ascending=False).head(n)
    return [{"home": r["home"], "away": r["away"],
             "p": {"home": float(r["p_home"]), "draw": float(r["p_draw"]),
                   "away": float(r["p_away"])},
             "xg": {"home": float(r["lam_home"]), "away": float(r["lam_away"])}}
            for _, r in ordered.iterrows()]


def build_run_message(cfg: Config) -> str:
    """Reconstruit le rapport Telegram à partir des JSON publiés (test manuel)."""
    d = cfg.site_dir / "data"

    def _load(name: str) -> dict:
        f = d / name
        if not f.exists():
            return {}
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    preds, led, hist = _load("predictions.json"), _load("ledger.json"), _load("history.json")
    settled = [b for b in led.get("bets", []) if b.get("status") in
               ("won", "lost", "push", "half_won", "half_lost")]
    hstats = hist.get("stats", {}) or {}
    tg = TelegramNotifier.from_config(cfg)
    return tg.daily_message(
        fixtures=len(preds.get("fixtures", [])), bets=preds.get("bets", []),
        settled=settled[-8:], stats=led.get("stats", {}), model=preds.get("model", {}),
        top=preds.get("fixtures", [])[:5], n_pred_archive=hstats.get("n", 0),
        hit_rate=hstats.get("hit_rate"))


def top_ratings(ctx: ModelContext, per_league: int = 4) -> dict:
    out: dict[str, list] = {}
    for code, model in ctx.ensemble.dc.items():
        table = model.team_strength().head(per_league)
        if table.empty:
            continue
        out[code] = [{"team": r.team, "rating": round(float(r.rating), 3),
                      "attack": round(float(r.attack), 3), "defense": round(float(r.defense), 3)}
                     for r in table.itertuples()]
    return out
