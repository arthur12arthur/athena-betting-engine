"""Le point le plus important du système : aucune fuite du futur."""
import numpy as np
import pandas as pd

from athena.etl.features import FeatureBuilder, prepare


def _fake_history(n=200, seed=0):
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(8)]
    rows = []
    start = pd.Timestamp("2023-08-01")
    for i in range(n):
        h, a = rng.choice(teams, 2, replace=False)
        hg, ag = int(rng.poisson(1.4)), int(rng.poisson(1.1))
        rows.append({
            "match_id": f"M{i}", "league": "E0", "season_start": 2023,
            "date": start + pd.Timedelta(days=3 * i), "home": h, "away": a,
            "home_goals": hg, "away_goals": ag,
            "result": "H" if hg > ag else ("D" if hg == ag else "A"),
            "xg_home": np.nan, "xg_away": np.nan, "sot_home": np.nan, "sot_away": np.nan,
        })
    return pd.DataFrame(rows)


def test_une_variable_ne_depend_pas_du_resultat_du_match():
    hist = _fake_history()
    train, _ = prepare(hist, None, {"E0": 0})
    # on modifie le résultat du DERNIER match : aucune feature de ce match
    # ne doit changer (elles sont calculées avant le coup d'envoi).
    hist2 = hist.copy()
    hist2.loc[hist2.index[-1], ["home_goals", "away_goals"]] = [9, 0]
    train2, _ = prepare(hist2, None, {"E0": 0})
    a = train.iloc[-1]
    b = train2.iloc[-1]
    for col in ["elo_home", "elo_away", "form5_home", "gf5_home", "rest_home"]:
        assert a[col] == b[col], col


def test_les_features_evoluent_apres_le_match():
    hist = _fake_history()
    train, _ = prepare(hist, None, {"E0": 0})
    team = train.iloc[5]["home"]
    later = train[(train["home"] == team) | (train["away"] == team)].iloc[1:]
    assert not later.empty
    # l'Elo d'une équipe évolue après ses matchs
    elos = train[(train["home"] == team)]["elo_home"].dropna()
    assert elos.nunique() > 1


def test_fixtures_utilisent_l_etat_final():
    hist = _fake_history(120)
    fixtures = pd.DataFrame([{
        "league": "E0", "date": hist["date"].max() + pd.Timedelta(days=3),
        "home": "T0", "away": "T1",
    }])
    train, tgt = prepare(hist, fixtures, {"E0": 0})
    assert len(tgt) == 1
    assert tgt.iloc[0]["elo_home"] > 0
    # l'Elo de la fixture = dernier Elo connu de l'équipe
    fb = FeatureBuilder()
    fb.build(hist, None)
    assert fb.elo.get("T0") == tgt.iloc[0]["elo_home"]
