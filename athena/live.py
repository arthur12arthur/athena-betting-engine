"""Pipeline quotidien ATHENA : ingestion -> règlement -> prédiction -> value -> publication.

Séquence exécutée chaque jour par GitHub Actions (et à la demande en CLI) :

  1. ``ingest``    : télécharge les résultats/cotes récents (football-data.co.uk)
  2. ``settle``    : règle les paris ouverts (résultats réels) et met à jour la bankroll
  3. ``fixtures``  : récupère le calendrier (ESPN) et les cotes (Pinnacle + OddsAPI)
  4. ``predict``   : modèle entraîné sur l'historique -> matrice de scores -> marchés
  5. ``select``    : value betting (edge vs prix sharp) + Kelly fractionnaire
  6. ``publish``   : écrit les JSON publics + le dashboard statique

Tout est idempotent : relancer deux fois le même jour ne duplique pas de pari.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime

import pandas as pd

from .betting.devig import blend_with_market, devig
from .betting.staking import plan_stake, settle_bet
from .betting.value import label
from .config import Config
from .etl.features import prepare
from .etl.names import TeamResolver
from .http import HttpClient
from .logging_conf import get_logger
from .models.ensemble import train
from .sources import football_data_csv as fd
from .sources.espn import fetch_recent_results, fetch_upcoming
from .sources.gemini import GeminiClient
from .sources.oddsapi import OddsApiClient
from .sources.pinnacle import PinnacleClient
from .state import STATUS_LOST, STATUS_PUSH, STATUS_WON, Ledger

log = get_logger("live")


# ==========================================================================
# Utilitaires
# ==========================================================================
def bet_id(league: str, kickoff: str, home: str, away: str, market: str, selection: str) -> str:
    raw = f"{league}|{kickoff[:10]}|{home}|{away}|{market}|{selection}"
    return hashlib.sha1(raw.encode()).hexdigest()[:14]


def _settle_market(market: str, selection: str, line: float | None, hg: int, ag: int) -> tuple[bool, bool]:
    if market == "1X2":
        res = "home" if hg > ag else ("draw" if hg == ag else "away")
        return selection == res, False
    if market == "DC":
        res = "home" if hg > ag else ("draw" if hg == ag else "away")
        wins = {"1X": res in {"home", "draw"}, "12": res in {"home", "away"}, "X2": res in {"draw", "away"}}
        return bool(wins.get(selection)), False
    if market == "OU25":
        return (selection == "over") == ((hg + ag) > 2.5), False
    if market == "BTTS":
        return (selection == "yes") == (hg > 0 and ag > 0), False
    if market == "AH":
        ln = float(line or -0.5)
        diff = (hg - ag) + ln
        if abs(diff) < 1e-9:
            return False, True
        return (diff > 0) if selection == "home" else (diff < 0), False
    return False, False


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
def settle_open_bets(cfg: Config, ledger: Ledger, history: pd.DataFrame,
                     espn_results: list | None = None) -> int:
    open_bets = ledger.open_bets()
    if not open_bets:
        return 0

    results: dict[tuple, tuple[int, int]] = {}
    for r in history.dropna(subset=["home_goals", "away_goals"]).itertuples():
        key = (str(r.league), str(pd.Timestamp(r.date).date()), r.home, r.away)
        results[key] = (int(r.home_goals), int(r.away_goals))
    for f in (espn_results or []):
        if f.status != "finished" or f.home_goals is None:
            continue
        d = str(pd.Timestamp(f.date_utc).date())
        results.setdefault((f.league, d, f.home, f.away), (int(f.home_goals), int(f.away_goals)))

    settled = 0
    for b in open_bets:
        ko = str(b["kickoff"])[:10]
        candidates = [
            (b["league"], ko, b["home"], b["away"]),
            (b["league"], str((pd.Timestamp(ko) + pd.Timedelta(days=1)).date()), b["home"], b["away"]),
            (b["league"], str((pd.Timestamp(ko) - pd.Timedelta(days=1)).date()), b["home"], b["away"]),
        ]
        score = next((results[c] for c in candidates if c in results), None)
        if score is None:
            # tolérance de nommage (ESPN vs canonical déjà normalisé en amont)
            alt = [k for k in results
                   if k[0] == b["league"] and k[2] == b["home"] and k[3] == b["away"]
                   and abs((pd.Timestamp(k[1]) - pd.Timestamp(ko)).days) <= 2]
            score = results[alt[0]] if alt else None
        if score is None:
            continue
        hg, ag = score
        won, push = _settle_market(b["market"], b["selection"], b.get("line"), hg, ag)
        pnl = settle_bet(float(b["stake"]), float(b["price"]), won, push)
        status = STATUS_PUSH if push else (STATUS_WON if won else STATUS_LOST)
        ledger.settle_bet(b["bet_id"], status, pnl, result=f"{hg}-{ag}",
                          closing_price=b.get("closing_price"))
        settled += 1
        log.info("réglé : %s %s (%s) -> %s %+.2f", b["home"], b["label"], f"{hg}-{ag}", status, pnl)
    if settled:
        ledger.save()
    return settled


# ==========================================================================
# 3. Fixtures + cotes
# ==========================================================================
@dataclass
class FixtureBoard:
    rows: list[dict] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def build_fixture_board(cfg: Config, client: HttpClient, resolver: TeamResolver,
                        horizon_days: int) -> FixtureBoard:
    """Fusionne calendrier (ESPN), cotes Pinnacle et cotes multi-books."""
    board: dict[tuple, dict] = {}

    def slot(league: str, date: str, home: str, away: str) -> dict:
        key = (league, home, away)
        if key not in board:
            board[key] = {"league": league, "date": date, "home": home, "away": away,
                          "sources": [], "pinnacle": {}, "pinnacle_odds": {}, "best_odds": {},
                          "books": {}, "n_books": 0}
        return board[key]

    # --- ESPN : calendrier
    for f in fetch_upcoming(cfg, client, horizon_days):
        home = resolver.resolve(f.home, f.league)
        away = resolver.resolve(f.away, f.league)
        if not home or not away:
            log.debug("non résolu ESPN: %s / %s (%s)", f.home, f.away, f.league)
            continue
        s = slot(f.league, str(pd.Timestamp(f.date_utc).date()), home, away)
        s["sources"].append("espn")
        s["kickoff_utc"] = f.date_utc

    # --- Pinnacle : cotes sharp
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
                s["pinnacle_totals"] = {str(k): v for k, v in snap.totals.items()}
                s["overround"] = round(snap.overround, 4)

    # --- OddsAPI : meilleures cotes multi-books (optionnel)
    oa = OddsApiClient(client=client)
    if oa.enabled:
        for lg in cfg.leagues:
            for ev in oa.odds_for_league(lg.code):
                home = resolver.resolve(ev["home"], lg.code)
                away = resolver.resolve(ev["away"], lg.code)
                if not home or not away:
                    continue
                date = str(pd.Timestamp(ev["commence"]).tz_convert("UTC").date()) if ev.get("commence") else ""
                s = slot(lg.code, date, home, away)
                s["sources"].append("oddsapi")
                s["best_odds"] = ev.get("best_1x2", {})
                s["books"] = ev.get("books", {})
                s["n_books"] = ev.get("n_books", 0)
                if ev.get("totals", {}).get(2.5):
                    s["best_totals"] = {"2.5": ev["totals"][2.5]}

    out = FixtureBoard(rows=[v for v in board.values() if v.get("kickoff_utc")])
    out.rows.sort(key=lambda r: (r["date"], r["league"]))
    log.info("board : %d matchs (%d avec cotes Pinnacle)", len(out.rows),
             sum(1 for r in out.rows if r["pinnacle_odds"]))
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
    """Entraîne l'ensemble sur l'historique, avec validation hors échantillon
    sur les 60 derniers jours (les scalaires de calibration ne sont pas
    ajustés sur les données d'entraînement)."""
    league_map = {lg.code: i for i, lg in enumerate(cfg.leagues)}
    feats, _ = prepare(history, None, league_map)
    cutoff = pd.Timestamp(history["date"].max()) - pd.Timedelta(days=60)
    tr = feats[feats["date"] <= cutoff]
    va = feats[feats["date"] > cutoff]
    if len(tr) < 3000:
        tr, va = feats, None
    ens = train(cfg, history[history["date"] <= (cutoff if va is not None else history["date"].max())], tr, va)

    teams: dict[str, list[str]] = {}
    for league, grp in history.groupby("league"):
        teams[str(league)] = sorted(set(grp["home"]) | set(grp["away"]))
    resolver = TeamResolver.from_config(cfg, teams_by_league=teams)
    return ModelContext(ensemble=ens, resolver=resolver, history=history, features=feats)


def predict_board(cfg: Config, ctx: ModelContext, board: FixtureBoard) -> pd.DataFrame:
    """Matrices de scores pour chaque fixture + features."""
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
        out.append({
            "league": feat["league"], "home": feat["home"], "away": feat["away"],
            "kickoff": str(pd.Timestamp(feat["date"]).date()),
            "p_home": p["home"], "p_draw": p["draw"], "p_away": p["away"],
            "p_ou25_over": mat.market("OU25")["over"],
            "p_btts_yes": mat.market("BTTS")["yes"],
            "p_dc_1x": mat.market("DC")["1X"],
            "lam_home": mat.lam_home, "lam_away": mat.lam_away,
            "top_scores": mat.top_scores(3),
            "matrix": mat,
            "elo_home": float(feat.get("elo_home", 0)), "elo_away": float(feat.get("elo_away", 0)),
            "form5_home": float(feat.get("form5_home", 0)), "form5_away": float(feat.get("form5_away", 0)),
            "xg5_home": float(feat.get("xg5_for_home", 0)), "xg5_away": float(feat.get("xg5_for_away", 0)),
        })
    return pd.DataFrame(out)


# ==========================================================================
# 5. Sélection des paris
# ==========================================================================
def select_bets(cfg: Config, preds: pd.DataFrame, board: FixtureBoard,
                ledger: Ledger) -> list[dict]:
    """Applique les règles de value betting et renvoie les paris à enregistrer."""
    if preds.empty:
        return []
    bcfg = cfg.betting
    lam = float(cfg.model.get("shrink_to_market", 0.70))
    board_idx = {(r["league"], r["home"], r["away"]): r for r in board.rows}

    bankroll = ledger.bankroll
    today = str(datetime.now(UTC).date())
    day_bets = [b for b in ledger.bets if str(b.get("created_at", ""))[:10] == today]
    day_exposure = sum(float(b["stake"]) for b in day_bets)
    max_exp = float(bcfg.get("max_daily_exposure_pct", 0.07)) * bankroll

    candidates: list[dict] = []
    for _, pr in preds.iterrows():
        row = board_idx.get((pr["league"], pr["home"], pr["away"]))
        if not row:
            continue
        pinn = row.get("pinnacle_odds") or {}
        best = row.get("best_odds") or {}
        if len(pinn) == 3:
            sharp = devig([pinn["home"], pinn["draw"], pinn["away"]], "power")
        elif len(best) == 3:
            sharp = devig([best["home"], best["draw"], best["away"]], "power")
        else:
            continue
        prices = best if len(best) == 3 else pinn
        if len(prices) != 3:
            continue
        p_model = {"home": pr["p_home"], "draw": pr["p_draw"], "away": pr["p_away"]}
        p_fair = blend_with_market(p_model, sharp, lam)
        has_soft = len(best) == 3 and row.get("n_books", 0) >= 3
        ss = bcfg.get("single_source", {}) or {}
        if not has_soft:
            if not ss.get("enabled", True):
                continue
            min_edge = float(ss.get("min_edge", 0.05))
            stake_scale = float(ss.get("stake_scale", 0.5))
        else:
            min_edge = float(bcfg.get("min_edge", 0.03))
            stake_scale = 1.0

        for sel in ("home", "draw", "away"):
            price = float(prices[sel])
            p = float(p_fair[sel])
            edge = p * price - 1.0
            if edge < min_edge:
                continue
            if edge > float(bcfg.get("max_edge", 0.10)):
                continue
            ratio = p / sharp[sel] if sharp[sel] > 0 else 1.0
            if not (float(bcfg.get("max_divergence_low", 0.75)) <= ratio
                    <= float(bcfg.get("max_divergence_high", 1.35))):
                continue
            if not (float(bcfg.get("min_odds", 1.45)) <= price <= float(bcfg.get("max_odds", 8.0))):
                continue
            if not (float(bcfg.get("min_prob", 0.06)) <= p <= float(bcfg.get("max_prob", 0.80))):
                continue
            candidates.append({
                "league": pr["league"], "home": pr["home"], "away": pr["away"],
                "kickoff": pr["kickoff"], "market": "1X2", "selection": sel,
                "label": label("1X2", sel), "price": round(price, 3),
                "p_model": round(float(p_model[sel]), 4), "p_fair": round(p, 4),
                "p_sharp": round(float(sharp[sel]), 4), "edge": round(edge, 4),
                "book": (row.get("books") or {}).get(sel, "pinnacle" if not has_soft else "?"),
                "n_books": row.get("n_books", 1),
                "multi_book": bool(has_soft),
                "stake_scale": stake_scale,
            })

    candidates.sort(key=lambda c: -c["edge"])
    chosen: list[dict] = []
    seen: set[str] = set()
    for c in candidates:
        if len(chosen) >= int(bcfg.get("max_bets_per_day", 8)):
            break
        key = f'{c["league"]}|{c["home"]}|{c["away"]}'
        if key in seen:
            continue
        if ledger.has_bet(bet_id(c["league"], c["kickoff"], c["home"], c["away"], c["market"], c["selection"])):
            continue
        remaining = max(0.0, max_exp - day_exposure)
        plan = plan_stake(c["p_fair"], c["price"], bankroll, bcfg, remaining_exposure=remaining)
        if plan.stake <= 0:
            continue
        scaled = round(plan.stake * float(c.get("stake_scale", 1.0)), 2)
        c.update({"stake": scaled, "kelly_full": plan.kelly_full,
                  "bet_id": bet_id(c["league"], c["kickoff"], c["home"], c["away"], c["market"], c["selection"]),
                  "created_at": datetime.now(UTC).isoformat(timespec="seconds")})
        day_exposure += scaled
        seen.add(key)
        chosen.append(c)
        if day_exposure >= max_exp:
            break
    log.info("value : %d paris retenus (sur %d candidats)", len(chosen), len(candidates))
    return chosen


# ==========================================================================
# 6. Publication
# ==========================================================================
def publish(cfg: Config, preds: pd.DataFrame, board: FixtureBoard, ledger: Ledger,
            new_bets: list[dict], model_info: dict, briefings: dict[str, str] | None = None) -> dict:
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
            "fair": {"home": round(1 / max(pr["p_home"], 1e-6), 2),
                     "draw": round(1 / max(pr["p_draw"], 1e-6), 2),
                     "away": round(1 / max(pr["p_away"], 1e-6), 2)},
            "ou25": round(pr["p_ou25_over"], 4), "btts": round(pr["p_btts_yes"], 4),
            "scores": [[s, round(p, 4)] for s, p in pr["top_scores"]],
            "elo": [round(pr["elo_home"]), round(pr["elo_away"])],
            "form": [round(pr["form5_home"], 2), round(pr["form5_away"], 2)],
        }
        row = next((r for r in board.rows if r["league"] == pr["league"]
                    and r["home"] == pr["home"] and r["away"] == pr["away"]), None)
        if row:
            if row.get("pinnacle_odds"):
                entry["sharp_odds"] = {k: round(v, 2) for k, v in row["pinnacle_odds"].items()}
            if row.get("best_odds"):
                entry["best_odds"] = {k: round(v, 2) for k, v in row["best_odds"].items()}
                entry["n_books"] = row.get("n_books", 0)
        key = f'{pr["league"]}:{pr["home"]}-{pr["away"]}'
        if key in briefings:
            entry["briefing"] = briefings[key]
        payload.append(entry)

    (data_dir / "predictions.json").write_text(json.dumps({
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": model_info, "fixtures": payload, "bets": new_bets,
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    (data_dir / "ledger.json").write_text(json.dumps({
        "stats": ledger.stats(), "bets": ledger.bets[-400:],
        "equity": ledger.data.get("equity", [])[-400:],
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    return {"n_fixtures": len(payload), "n_bets": len(new_bets)}


# ==========================================================================
# Orchestration
# ==========================================================================
def run_daily(cfg: Config, do_publish: bool = True, use_llm: bool | None = None) -> dict:
    started = datetime.now(UTC)
    client = HttpClient(cache_dir=cfg.cache_dir)
    ledger_path = cfg.root / "data" / "state" / "ledger.json"
    ledger = Ledger.load(ledger_path, float(cfg.betting.get("initial_bankroll", 1000.0)))

    log.info("=== ATHENA : exécution quotidienne ===")
    history = ingest(cfg, client)
    n_settled = settle_open_bets(cfg, ledger, history, espn_results=fetch_recent_results(cfg, client, 4))

    ctx = build_model(cfg, history)
    horizon = int(cfg.publish.get("horizon_days", 3))
    board = build_fixture_board(cfg, client, ctx.resolver, horizon)
    preds = predict_board(cfg, ctx, board)

    # CLV : on mémorise la dernière cote sharp vue avant le coup d'envoi.
    # Au règlement, CLV = cote jouée / cote de clôture - 1.
    updated = 0
    for b in ledger.open_bets():
        for row in board.rows:
            if (row["league"], row["home"], row["away"]) == (b["league"], b["home"], b["away"]):
                price = (row.get("pinnacle_odds") or {}).get(b["selection"])
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

    # briefings LLM (optionnels, sur les paris uniquement)
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
                    value_note=f"Value détectée : {b['label']} à {b['price']} (edge {b['edge']:+.1%}).",
                )
                if txt:
                    briefings[f'{b["league"]}:{b["home"]}-{b["away"]}'] = txt

    ens = ctx.ensemble
    model_info = {
        "version": ens.version,
        "trained_at": ens.trained_at,
        "w_dc": round(ens.w_dc, 3),
        "temperature": round(ens.temperature, 3),
        "calibration": ens.metrics,
        "gbm_val": ens.gbm.metrics if ens.gbm else {},
        "n_matches_train": int(len(history)),
        "n_leagues": len(ens.dc),
        "features": 57,
        "leagues": [lg.code for lg in cfg.leagues],
    }

    out = {"fixtures": len(preds), "bets": len(new_bets), "settled": n_settled,
           "bankroll": ledger.bankroll, "model": model_info["version"]}
    if do_publish:
        out.update(publish(cfg, preds, board, ledger, new_bets, model_info, briefings))
        ratings = top_ratings(ctx)
        (cfg.site_dir / "data" / "ratings.json").write_text(
            json.dumps(ratings, ensure_ascii=False, indent=1), encoding="utf-8")
    ledger.log_run({"fixtures": len(preds), "bets": len(new_bets), "settled": n_settled,
                    "duration_s": round((datetime.now(UTC) - started).total_seconds(), 1)})
    ledger.save()
    log.info("=== terminé en %.1fs : %d matchs, %d paris, %d réglés, bankroll %.2f ===",
             (datetime.now(UTC) - started).total_seconds(),
             len(preds), len(new_bets), n_settled, ledger.bankroll)
    return out


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
