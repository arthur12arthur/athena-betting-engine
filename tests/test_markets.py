import pytest

from athena.models.markets import ScoreMatrix


def test_matrix_normalisee():
    m = ScoreMatrix.from_lambdas(1.5, 1.1, 8)
    assert m.mat.sum() == pytest.approx(1.0)
    p = m.probs_1x2()
    assert sum(p.values()) == pytest.approx(1.0)
    assert p["home"] > p["away"]


def test_avantage_du_terrain():
    m = ScoreMatrix.from_lambdas(1.6, 1.0, 8)
    m2 = ScoreMatrix.from_lambdas(1.0, 1.6, 8)
    assert m.probs_1x2()["home"] > m2.probs_1x2()["home"]


def test_reconcile_exact():
    m = ScoreMatrix.from_lambdas(1.4, 1.3, 8)
    r = m.reconcile_to(0.5, 0.25, 0.25)
    p = r.probs_1x2()
    assert p["home"] == pytest.approx(0.5, abs=1e-9)
    assert p["draw"] == pytest.approx(0.25, abs=1e-9)
    assert p["away"] == pytest.approx(0.25, abs=1e-9)
    assert r.mat.sum() == pytest.approx(1.0)


def test_marches_coherents():
    m = ScoreMatrix.from_lambdas(1.7, 1.2, 10)
    assert sum(m.market("OU25").values()) == pytest.approx(1.0)
    assert sum(m.market("BTTS").values()) == pytest.approx(1.0)
    dc = m.market("DC")
    p = m.probs_1x2()
    assert dc["1X"] == pytest.approx(p["home"] + p["draw"])
    assert dc["X2"] == pytest.approx(p["draw"] + p["away"])
    ah = m.market("AH", -0.5)
    assert ah["home"] == pytest.approx(p["home"])
    ah0 = m.market("AH", 0.0)
    assert ah0["push"] == pytest.approx(p["draw"])


def test_blend_convexe():
    a = ScoreMatrix.from_lambdas(2.0, 0.8, 8)
    b = ScoreMatrix.from_lambdas(0.8, 2.0, 8)
    mid = a.blend(b, 0.5)
    assert mid.probs_1x2()["home"] == pytest.approx(
        (a.probs_1x2()["home"] + b.probs_1x2()["home"]) / 2, abs=1e-9)
    assert mid.mat.sum() == pytest.approx(1.0)


def test_sharpen_respecte_somme():
    m = ScoreMatrix.from_lambdas(1.4, 1.2, 8)
    s = m.sharpen(0.8)
    assert s.mat.sum() == pytest.approx(1.0)
    assert s.probs_1x2()["home"] >= m.probs_1x2()["home"]
