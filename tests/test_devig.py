import pytest

from athena.betting.devig import (
    additive,
    blend_with_market,
    devig,
    fair_odds,
    multiplicative,
    overround,
    power,
    shin,
)

ODDS = [2.30, 3.95, 2.50]


@pytest.mark.parametrize("method", ["multiplicative", "additive", "power", "shin"])
def test_probabilites_normalisees(method):
    p = devig(ODDS, method)
    assert sum(p.values()) == pytest.approx(1.0, abs=1e-9)
    assert all(0 < v < 1 for v in p.values())


def test_surround_retire():
    assert overround(ODDS) > 1.0
    p = multiplicative(ODDS)
    assert p.sum() == pytest.approx(1.0)


def test_favori_dans_toutes_les_methodes():
    for fn in (multiplicative, additive, power, shin):
        p = fn(ODDS)
        assert p[0] > p[1] and p[0] > p[2]


def test_prix_juste():
    assert fair_odds(0.5) == pytest.approx(2.0, abs=0.01)
    assert fair_odds(0.5, margin=0.02) < 2.0


def test_blend_avec_marche():
    model = {"home": 0.5, "draw": 0.3, "away": 0.2}
    market = {"home": 0.4, "draw": 0.3, "away": 0.3}
    mix = blend_with_market(model, market, 0.5)
    assert mix["home"] == pytest.approx(0.45)
    assert sum(mix.values()) == pytest.approx(1.0)
    assert blend_with_market(model, market, 0.0)["home"] == pytest.approx(0.5)
