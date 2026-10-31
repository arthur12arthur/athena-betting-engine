import pytest

from athena.betting.staking import kelly_fraction, plan_stake, settle_bet

CFG = {"kelly_fraction": 0.25, "max_stake_pct": 0.02, "min_stake_pct": 0.001}


def test_kelly_positif_seulement_avec_edge():
    assert kelly_fraction(0.5, 2.0) == pytest.approx(0.0)
    assert kelly_fraction(0.55, 2.0) == pytest.approx(0.10)
    assert kelly_fraction(0.4, 2.0) == 0.0


def test_plafond_par_pari():
    plan = plan_stake(0.9, 5.0, 1000, CFG)
    assert plan.stake == pytest.approx(20.0)
    assert plan.capped_by == "max_stake_pct"


def test_exposition_journaliere():
    plan = plan_stake(0.6, 3.0, 1000, CFG, remaining_exposure=3.0)
    assert plan.stake <= 3.0


def test_mise_minimale_rejetee():
    plan = plan_stake(0.501, 2.0, 1000, CFG)
    assert plan.stake == 0.0


def test_reglement():
    assert settle_bet(10, 2.5, True) == pytest.approx(15.0)
    assert settle_bet(10, 2.5, False) == pytest.approx(-10.0)
    assert settle_bet(10, 2.5, False, push=True) == 0.0
