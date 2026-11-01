"""Moteur de backtest walk-forward (aucune fuite du futur).

Pour chaque saison testée S :
  - entraînement : toutes les saisons < S
  - validation    : saison S-1 (poids de l'ensemble + température)
  - test          : saison S, jour par jour, dans l'ordre chronologique

Cotes utilisées :
  - on parie aux cotes **pré-clôture moyennes du marché** (Avg), dégradées
    d'un slippage (1,5 %) : c'est pessimiste et réaliste.
  - la référence de probabilité est la clôture **sharp** (Pinnacle) dé-viggée.
  - le CLV mesure l'écart entre notre cote et la clôture sharp.

Deux temps : ``prepare`` (coûteux : entraînement) puis ``simulate`` (rapide),
ce qui permet de balayer les paramètres de mise sans ré-entraîner.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..betting.devig import blend_with_market
from ..betting.offers import Offer, offers_from_history
from ..betting.settlement import settle_pnl
from ..betting.staking import plan_stake
from ..betting.value import label
from ..logging_conf import get_logger
from ..models.ensemble import train
from ..models.markets import parse_market
from . import metrics

log = get_logger("backtest")


# --------------------------------------------------------------------------
# Règlement
# --------------------------------------------------------------------------
def settle(market: str, selection: str, line: float | None, hg: int, ag: int) -> tuple[bool, bool]:
    """Renvoie (gagné, push) pour un pari."""
    if market == "1X2":
        res = "home" if hg > ag else ("draw" if hg == ag else "away")
        return selection == res, False
    if market == "DC":
        res = "home" if hg > ag else ("draw" if hg == ag else "away")
        wins = {"1X": res in {"home", "draw"}, "12": res in {"home", "away"},
                "X2": res in {"draw", "away"}}
        return bool(wins.get(selection, False)), False
    if market == "OU25":
        over = (hg + ag) > 2.5
        return (selection == "over") == over, False
    if market == "BTTS":
        yes = hg > 0 and ag > 0
        return (selection == "yes") == yes, False
    if market == "AH":
        ln = float(line or -0.5)
        diff = (hg - ag) + ln
        if abs(diff) < 1e-9:
            return False, True
        return (diff > 0) if selection == "home" else (diff < 0), False
    return False, False


# --------------------------------------------------------------------------
# Préparation (entraînement) d'une saison
# --------------------------------------------------------------------------
@dataclass
class SeasonBundle:
    season: int
    rows: pd.DataFrame
    matrices: list
    offers: list[list[Offer]]
    ensemble_info: dict = field(default_factory=dict)


def prepare_season(cfg, history: pd.DataFrame, feats: pd.DataFrame, season: int,
                   min_train_matches: int = 4000) -> SeasonBundle | None:
    # validation hors échantillon : la saison S-1 n'entre PAS dans l'entraînement
    tr = feats[feats["season_start"] < season - 1]
    va = feats[feats["season_start"] == season - 1]
    te = feats[feats["season_start"] == season].dropna(subset=["result_code"]).copy()
    if len(tr) < min_train_matches or te.empty:
        log.warning("saison %s ignorée (train=%d test=%d)", season, len(tr), len(te))
        return None
    te["result_code"] = te["result_code"].astype(int)
    te = te.sort_values("date").reset_index(drop=True)
    hist = history[history["date"] <= pd.Timestamp(te["date"].min())]
    ens = train(cfg, hist, tr, va)
    mats = ens.predict_frame(te)
    markets = list(cfg.betting.get("markets", ["1X2"]))
    price_mode = str(cfg.betting.get("price_mode", "avg"))
    offers = [offers_from_history(r, markets, price_mode) for _, r in te.iterrows()]
    return SeasonBundle(
        season=season, rows=te, matrices=mats, offers=offers,
        ensemble_info={"w_dc": ens.w_dc, "temperature": ens.temperature,
                       "version": ens.version, "metrics": ens.metrics,
                       "gbm": ens.gbm.metrics if ens.gbm else {}},
    )


# --------------------------------------------------------------------------
# Simulation des paris
# --------------------------------------------------------------------------
def _market_params(bcfg: dict, market: str) -> dict:
    family, _ = parse_market(market)
    overrides = (bcfg.get("market_params") or {}).get(market) or \
                (bcfg.get("market_params") or {}).get(family) or {}
    return {**bcfg, **overrides}


def simulate(bundle: SeasonBundle, betting_cfg: dict, initial_bankroll: float,
             lambda_market: float = 0.35) -> tuple[pd.DataFrame, list[dict], list[float]]:
    """Rejoue la saison avec les règles de mise fournies (multi-marchés)."""
    bankroll = float(initial_bankroll)
    bets: list[dict] = []
    day_pnl: list[dict] = []

    max_day = int(betting_cfg.get("max_bets_per_day", 8))
    max_match = int(betting_cfg.get("max_bets_per_match", 1))
    max_exp_pct = float(betting_cfg.get("max_daily_exposure_pct", 0.07))
    slippage = float(betting_cfg.get("slippage", 0.015))
    active = set(betting_cfg.get("markets", ["1X2"]))

    df = bundle.rows.copy()
    df["_day"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    for day, group in df.groupby("_day", sort=True):
        exposure = 0.0
        candidates: list[tuple[float, int, Offer, float, dict]] = []

        for idx in group.index:
            row = df.loc[idx]
            mat = bundle.matrices[idx]
            hg, ag = int(row["home_goals"]), int(row["away_goals"])
            pools: dict[tuple[str, float | None], list[Offer]] = {}
            for o in bundle.offers[idx]:
                if o.market not in active and parse_market(o.market)[0] not in active:
                    continue
                pools.setdefault((o.market, o.line), []).append(o)
            for (market, line), group_offers in pools.items():
                params = _market_params(betting_cfg, market)
                family, ln = parse_market(market, line)
                try:
                    p_model = mat.market(family, ln)
                except Exception:  # noqa: BLE001
                    continue
                p_sharp = {o.selection: o.p_sharp for o in group_offers if o.p_sharp}
                if not p_sharp:
                    continue
                p_fair = blend_with_market(p_model, p_sharp, lambda_market)
                for o in group_offers:
                    p = p_fair.get(o.selection)
                    if p is None:
                        continue
                    price = float(o.price_bet)
                    edge = p * price - 1.0
                    if edge < float(params.get("min_edge", 0.03)):
                        continue
                    if edge > float(params.get("max_edge", 0.10)):
                        continue
                    ratio = p / o.p_sharp if o.p_sharp else 1.0
                    if not (float(betting_cfg.get("max_divergence_low", 0.75)) <= ratio
                            <= float(betting_cfg.get("max_divergence_high", 1.35))):
                        continue
                    if not (float(params.get("min_odds", 1.45)) <= price
                            <= float(params.get("max_odds", 8.0))):
                        continue
                    if not (float(params.get("min_prob", 0.06)) <= p
                            <= float(params.get("max_prob", 0.80))):
                        continue
                    candidates.append((float(edge), idx, o, float(p), {"hg": hg, "ag": ag,
                                                                      "row": row,
                                                                      "p_model": p_model}))

        candidates.sort(key=lambda c: -c[0])
        per_match: dict[int, int] = {}
        n_day = 0
        for edge, idx, o, p_fair, ctx in candidates:
            if n_day >= max_day:
                break
            if per_match.get(idx, 0) >= max_match:
                continue
            remaining = max(0.0, max_exp_pct * bankroll - exposure)
            price = round(o.price_bet * (1.0 - slippage), 3)
            plan = plan_stake(p_fair, price, bankroll, betting_cfg, remaining_exposure=remaining)
            if plan.stake <= 0:
                continue
            hg, ag = int(ctx["hg"]), int(ctx["ag"])
            pnl, status = settle_pnl(o.market, o.selection, o.line, plan.stake, price, hg, ag)
            bankroll += pnl
            exposure += plan.stake
            n_day += 1
            per_match[idx] = per_match.get(idx, 0) + 1
            row = ctx["row"]
            family, ln = parse_market(o.market, o.line)
            bets.append({
                "season": bundle.season, "kickoff": day, "league": row["league"],
                "home": row["home"], "away": row["away"], "match_index": int(idx),
                "market": o.market, "selection": o.selection, "line": ln,
                "label": label(family, o.selection, ln),
                "price": price, "p_model": round(float(ctx["p_model"].get(o.selection, 0.0)), 4),
                "p_fair": round(p_fair, 4), "p_sharp": round(float(o.p_sharp or 0.0), 4),
                "edge": round(edge, 4), "stake": plan.stake, "pnl": pnl, "status": status,
                "result": f"{hg}-{ag}", "clv": o.clv(),
                "bankroll_after": round(bankroll, 2),
            })
        day_pnl.append({"date": day, "bankroll": round(bankroll, 2), "n": n_day})

    return pd.DataFrame(bets), day_pnl, []


# --------------------------------------------------------------------------
# Rapport complet
# --------------------------------------------------------------------------
def evaluate(bundles: list[SeasonBundle], betting_cfg: dict, initial_bankroll: float,
             lambda_market: float = 0.35) -> dict:
    all_bets: list[pd.DataFrame] = []
    P, y = [], []
    bankroll = initial_bankroll
    per_season = []
    for b in bundles:
        bets, _days, _ = simulate(b, betting_cfg, bankroll, lambda_market)
        if len(bets):
            bankroll = float(bets["bankroll_after"].iloc[-1])
        all_bets.append(bets)
        rows = b.rows
        for i, mat in enumerate(b.matrices):
            p = mat.probs_1x2()
            P.append([p["home"], p["draw"], p["away"]])
            y.append(int(rows.iloc[i]["result_code"]))
        sd = metrics.performance(bets, initial_bankroll)
        per_season.append({"season": b.season, "n_bets": sd["n_bets"], "pnl": sd["pnl"],
                           "roi": sd["roi"], "bankroll_final": sd["bankroll"],
                           **b.ensemble_info})
    bets_df = pd.concat(all_bets, ignore_index=True) if all_bets else pd.DataFrame()
    Pa, ya = np.array(P), np.array(y)
    perf = metrics.performance(bets_df, initial_bankroll) if len(bets_df) else {"n_bets": 0}
    report = {
        "generated_at": pd.Timestamp.utcnow().isoformat(timespec="seconds"),
        "seasons": [b.season for b in bundles],
        "initial_bankroll": initial_bankroll,
        "predictions": metrics.summary_predictions(Pa, ya) if len(ya) else {},
        "performance": perf,
        "per_season": per_season,
        "by_league": metrics.breakdown(bets_df, "league", initial_bankroll) if len(bets_df) else [],
        "by_market": metrics.breakdown(bets_df, "market", initial_bankroll) if len(bets_df) else [],
        "bets": bets_df.tail(400).to_dict(orient="records") if len(bets_df) else [],
        "equity_curve": perf.get("equity_curve", []),
    }
    return report


def run(cfg, history: pd.DataFrame, feats: pd.DataFrame, test_seasons: list[int] | None = None,
        initial_bankroll: float | None = None, lambda_market: float | None = None) -> dict:
    seasons = sorted(feats["season_start"].dropna().unique().astype(int).tolist())
    test_seasons = test_seasons or seasons[-2:]
    bankroll = float(initial_bankroll or cfg.betting.get("initial_bankroll", 1000.0))
    lam = float(lambda_market if lambda_market is not None
                else cfg.model.get("shrink_to_market", 0.15))
    log.info("backtest walk-forward : saisons %s | bankroll %.0f | λ_marché %.2f",
             test_seasons, bankroll, lam)
    bundles = [b for s in test_seasons if (b := prepare_season(cfg, history, feats, s))]
    if not bundles:
        return {"error": "aucune saison exploitable"}
    return evaluate(bundles, cfg.betting, bankroll, lam)
