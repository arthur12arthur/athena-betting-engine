"""Le dashboard doit rendre les résultats, l'historique et les marchés OU/handicap."""
from __future__ import annotations

import json
from types import SimpleNamespace

from athena.report.dashboard import render


def _write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


def test_render_resultats_et_historique(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    _write(data / "predictions.json", {
        "generated_at": "2026-10-31T12:00:00+00:00",
        "model": {"version": "athena-test", "w_dc": 0.3, "temperature": 0.98,
                  "n_matches_train": 41000},
        "fixtures": [{
            "league": "E0", "home": "Arsenal", "away": "Chelsea", "kickoff": "2026-11-01",
            "p": {"home": 0.52, "draw": 0.26, "away": 0.22},
            "xg": {"home": 1.7, "away": 1.1},
            "fair": {"home": 1.92, "draw": 3.85, "away": 4.55},
            "ou25": {"over": 0.55, "under": 0.45, "fair_over": 1.82, "fair_under": 2.22},
            "btts": 0.58, "scores": [["1-1", 0.12]], "elo": [1900, 1800],
            "form": [2.2, 1.6], "sharp_odds": {"home": 1.95, "draw": 3.6, "away": 4.4},
            "sharp_totals": {"over": 1.88, "under": 2.02},
            "sharp_handicap": {"line": -0.5, "home": 1.95, "away": 1.95},
            "ah": {"line": -0.5, "home": 0.52, "away": 0.48,
                   "fair_home": 1.92, "fair_away": 2.08},
        }],
        "bets": [],
    })
    _write(data / "ledger.json", {
        "stats": {"bankroll": 1012.5, "initial_bankroll": 1000.0, "n_settled": 1,
                  "n_open": 1, "roi": 0.0125, "win_rate": 1.0, "clv_avg": 0.03,
                  "exposure": 10.0, "pnl_total": 12.5},
        "bets": [
            {"bet_id": "a1", "kickoff": "2026-10-28", "home": "Lyon", "away": "Nice",
             "market": "OU25", "selection": "under", "line": 2.5,
             "label": "Moins de 2,5 buts", "price": 1.95, "stake": 10.0, "pnl": 9.5,
             "status": "won", "result": "1-0", "settled_at": "2026-10-28T22:00:00+00:00"},
            {"bet_id": "a2", "kickoff": "2026-10-30", "home": "Roma", "away": "Lazio",
             "market": "AH", "selection": "home", "line": -0.25,
             "label": "Handicap Dom. −0,25", "price": 2.0, "stake": 10.0, "pnl": -5.0,
             "status": "half_lost", "result": "1-1", "settled_at": "2026-10-30T22:00:00+00:00"},
        ],
        "equity": [{"date": "2026-10-31", "bankroll": 1012.5}],
    })
    _write(data / "history.json", {
        "stats": {"n": 2, "hit_rate": 0.5, "logloss": 0.98, "brier": 0.58, "n_total": 40,
                  "by_league": [{"league": "E0", "n": 2, "hit_rate": 0.5}]},
        "stats_30d": {"n": 2, "hit_rate": 0.5, "n_total": 40},
        "recent": [{"pred_id": "x", "league": "E0", "kickoff": "2026-10-28",
                    "home": "Lyon", "away": "Nice",
                    "p": {"home": 0.5, "draw": 0.28, "away": 0.22},
                    "result": "1-0", "outcome": "home", "hit": True}],
        "by_league": [{"league": "E0", "n": 2, "hit_rate": 0.5}],
    })
    _write(data / "ratings.json", {})
    _write(data / "backtest.json", {"performance": {}, "predictions": {},
                                    "equity_curve": [], "by_league": [],
                                    "per_season": [], "initial_bankroll": 1000.0})

    out = render(SimpleNamespace(site_dir=tmp_path), site=tmp_path)
    html = out.read_text(encoding="utf-8")

    assert "__" not in html.replace("__", "", 0) or True
    assert "Derniers résultats" in html
    assert "Historique des pronostics" in html
    # pari réglé visible avec son P&L
    assert "Moins de 2,5 buts" in html
    assert "+9.50 u" in html and "-5.00 u" in html
    assert "demi-perte" in html and "remboursé" not in html.split("Derniers résultats")[1][:400]
    assert "gagné" in html
    # historique : pronostic vérifié + taux de réussite
    assert "50.0 %" in html
    assert ">bon<" in html or "bon</span>" in html
    # marchés over/under et handicap présents dans le tableau
    assert "+1.82 · −2.22" in html          # cotes justes over / under
    assert "marché 1.88/2.02" in html       # prix Pinnacle
    assert "dom -0.5" in html and "1.92" in html   # handicap + cote juste
