"""Règlement multi-marchés, offres et historique des pronostics."""
from __future__ import annotations

import pytest

from athena.betting.offers import Offer, offers_from_history, offers_from_live
from athena.betting.settlement import SETTLED_STATUSES, settle_bool, settle_pnl
from athena.history import PredictionArchive, pred_id


# ------------------------------------------------------------------ règlement
@pytest.mark.parametrize("market,selection,line,hg,ag,expected_pnl,expected_status", [
    ("1X2", "home", None, 2, 0, 10.0, "won"),
    ("1X2", "away", None, 2, 0, -10.0, "lost"),
    ("1X2", "draw", None, 1, 1, 10.0, "won"),
    ("DC", "1X", None, 1, 1, 10.0, "won"),
    ("DC", "X2", None, 2, 1, -10.0, "lost"),
    ("OU2.5", "over", 2.5, 2, 1, 10.0, "won"),
    ("OU2.5", "under", 2.5, 2, 1, -10.0, "lost"),
    ("OU3", "over", 3.0, 2, 1, 0.0, "push"),
    ("OU3.25", "over", 3.25, 2, 1, -5.0, "half_lost"),
    ("OU3.25", "under", 3.25, 2, 1, 5.0, "half_won"),
    ("AH", "home", -0.5, 1, 0, 10.0, "won"),
    ("AH", "home", -0.75, 1, 0, 5.0, "half_won"),
    ("AH", "away", -0.75, 0, 0, -10.0, "lost"),
    ("AH", "away", 0.75, 1, 1, 10.0, "won"),
    ("AH", "home", 0.25, 1, 1, 5.0, "half_won"),
    ("AH", "home", -0.75, 1, 1, -10.0, "lost"),
    ("AH", "home", 0.0, 1, 1, 0.0, "push"),
    ("AH", "away", 0.0, 1, 1, 0.0, "push"),
    ("AH", "away", 0.25, 1, 1, 5.0, "half_won"),
    ("AH", "home", -0.25, 1, 1, -5.0, "half_lost"),
])
def test_settle_pnl(market, selection, line, hg, ag, expected_pnl, expected_status):
    pnl, status = settle_pnl(market, selection, line, 10.0, 2.0, hg, ag)
    assert status == expected_status
    assert pnl == pytest.approx(expected_pnl)
    assert status in SETTLED_STATUSES


def test_settle_bool_compat():
    assert settle_bool("1X2", "home", None, 1, 0) == (True, False)
    assert settle_bool("1X2", "home", None, 0, 1) == (False, False)
    assert settle_bool("AH", "home", 0.0, 1, 1) == (False, True)


# --------------------------------------------------------------------- offres
ROW = {
    "league": "E0", "date": "2026-10-01", "home": "A", "away": "B",
    "odds_avg_home": 2.10, "odds_avg_draw": 3.40, "odds_avg_away": 3.60,
    "odds_psc_home": 2.05, "odds_psc_draw": 3.45, "odds_psc_away": 3.70,
    "odds_ou_avg_over": 1.95, "odds_ou_avg_under": 1.90,
    "odds_ou_psc_over": 1.97, "odds_ou_psc_under": 1.88,
    "odds_ah_line": -0.5, "odds_ah_avg_home": 1.90, "odds_ah_avg_away": 1.95,
    "odds_ah_psc_home": 1.92, "odds_ah_psc_away": 1.93,
}


def test_parse_market_ligne_explicite():
    from athena.models.markets import parse_market
    assert parse_market("OU25") == ("OU", 2.5)
    assert parse_market("OU25", 2.25) == ("OU", 2.25)
    assert parse_market("OU2.5") == ("OU", 2.5)
    assert parse_market("AH", -0.75) == ("AH", -0.75)
    assert parse_market("1X2") == ("1X2", None)


def test_offers_from_history_multi_marches():
    offers = offers_from_history(ROW, ["1X2", "OU25", "AH"], price_mode="avg")
    markets = {o.market for o in offers}
    assert markets == {"1X2", "OU25", "AH"}
    for o in offers:
        assert o.price_bet > 1.0
        assert 0.0 < o.p_sharp < 1.0
    # les probabilités sharp de chaque marché somment à ~1
    for mk in markets:
        grp = [o for o in offers if o.market == mk]
        assert sum(o.p_sharp for o in grp) == pytest.approx(1.0, abs=1e-6)


def test_offers_from_live_prefere_le_meilleur_prix():
    row = {
        "pinnacle_odds": {"home": 2.00, "draw": 3.50, "away": 3.80},
        "pinnacle_totals": {"2.5": {"over": 1.90, "under": 1.95}},
        "pinnacle_handicaps": {"-0.5": {"home": 1.92, "away": 1.93}},
        "best_odds": {"home": 2.12, "draw": 3.60, "away": 3.90},
        "best_totals": {"2.5": {"over": 2.05, "under": 1.98}},
        "best_handicaps": {"-0.5": {"home": 2.00, "away": 2.02}},
        "n_books": 5,
    }
    offers = offers_from_live(row, ["1X2", "OU25", "AH"], min_books=3)
    home = next(o for o in offers if o.market == "1X2" and o.selection == "home")
    assert home.price_bet == pytest.approx(2.12)          # meilleur prix jouable
    # probabilité dé-viggée de Pinnacle (2.00 / 3.50 / 3.80 -> ~0,48)
    assert 0.45 < home.p_sharp < 0.52
    assert home.price_sharp == pytest.approx(2.00)
    assert home.clv() is not None and home.clv() > 0
    over = next(o for o in offers if o.market == "OU25" and o.selection == "over")
    assert over.line == pytest.approx(2.5)
    ah = next(o for o in offers if o.market == "AH")
    assert ah.line == pytest.approx(-0.5)


def test_offer_clv_sign():
    o = Offer("1X2", "home", None, 2.20, 2.00, 0.50)
    assert o.clv() == pytest.approx(0.10)
    assert Offer("1X2", "home", None, 1.90, 2.00, 0.50).clv() < 0
    assert Offer("1X2", "home", None, 1.90, None, None).clv() is None


# ------------------------------------------------------------------ historique
def test_archive_cycle_complet(tmp_path):
    path = tmp_path / "preds.json"
    arch = PredictionArchive.load(path)
    assert arch.stats()["n"] == 0
    preds = [
        {"league": "E0", "kickoff": "2026-10-01", "home": "A", "away": "B",
         "p_home": 0.60, "p_draw": 0.25, "p_away": 0.15, "lam_home": 2.0, "lam_away": 0.8},
        {"league": "E0", "kickoff": "2026-10-02", "home": "C", "away": "D",
         "p_home": 0.20, "p_draw": 0.30, "p_away": 0.50, "lam_home": 0.9, "lam_away": 1.7},
    ]
    assert arch.add_many(preds) == 2
    assert arch.add_many(preds) == 0          # idempotent
    n = arch.settle({("E0", "2026-10-01", "A", "B"): (2, 0),
                     ("E0", "2026-10-02", "C", "D"): (0, 1)})
    assert n == 2
    st = arch.stats()
    assert st["n"] == 2 and st["hit_rate"] == 1.0
    arch.save()

    reloaded = PredictionArchive.load(path)
    assert len(reloaded.entries) == 2
    assert reloaded.stats()["n"] == 2
    recent = reloaded.recent(5)
    assert recent[0]["result"] == "0-1" and recent[0]["outcome"] == "away"
    assert recent[0]["hit"] is True


def test_pred_id_stable():
    a = pred_id("E0", "2026-10-01 19:00", "A", "B")
    b = pred_id("E0", "2026-10-01", "A", "B")
    assert a == b and len(a) == 14
    assert a != pred_id("E0", "2026-10-01", "B", "A")
