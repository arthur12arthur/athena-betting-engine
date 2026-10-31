import numpy as np
import pandas as pd
import pytest

from athena.models.dixon_coles import DixonColesModel


def _synthetic(strong="Fort", weak="Faible", n=140, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    start = pd.Timestamp("2023-08-01")
    for i in range(n):
        if i % 2 == 0:
            home, away = strong, weak
        else:
            home, away = weak, strong
        lam_h = 2.2 if home == strong else 0.7
        lam_a = 2.0 if away == strong else 0.6
        rows.append({
            "match_id": f"M{i}", "league": "X", "season_start": 2023,
            "date": start + pd.Timedelta(days=2 * i), "home": home, "away": away,
            "home_goals": int(rng.poisson(lam_h)), "away_goals": int(rng.poisson(lam_a)),
        })
    # remplissage : autres équipes moyennes pour stabiliser l'estimation
    for j in range(n // 2):
        rows.append({
            "match_id": f"N{j}", "league": "X", "season_start": 2023,
            "date": start + pd.Timedelta(days=2 * j + 1), "home": "Moyen1", "away": "Moyen2",
            "home_goals": int(rng.poisson(1.4)), "away_goals": int(rng.poisson(1.3)),
        })
    return pd.DataFrame(rows)


def test_detecte_l_equipe_forte():
    df = _synthetic()
    model = DixonColesModel.fit(df)
    assert model is not None
    table = model.team_strength()
    assert table.iloc[0]["team"] == "Fort"
    lh, la = model.lambdas("Fort", "Faible")
    assert lh > la
    assert model.home_adv > 0


def test_probabilites_valides():
    model = DixonColesModel.fit(_synthetic())
    p = model.matrix("Fort", "Faible").probs_1x2()
    assert sum(p.values()) == pytest.approx(1.0)
    assert p["home"] > 0.5


def test_equipe_inconnue_traitee_comme_moyenne():
    model = DixonColesModel.fit(_synthetic())
    lh, la = model.lambdas("Inconnu", "Faible")
    assert 0.05 < lh < 8.0 and 0.05 < la < 8.0


def test_refuse_trop_peu_de_donnees():
    assert DixonColesModel.fit(_synthetic(n=20)) is None
