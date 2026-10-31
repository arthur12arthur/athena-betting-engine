import json

import pytest

from athena.state import STATUS_LOST, STATUS_WON, Ledger


def _ledger(tmp_path):
    return Ledger.load(tmp_path / "ledger.json", 1000.0)


def test_creation_et_persistance(tmp_path):
    led = _ledger(tmp_path)
    assert led.bankroll == 1000.0
    assert (tmp_path / "ledger.json").exists()
    json.loads((tmp_path / "ledger.json").read_text())


def test_ajout_idempotent(tmp_path):
    led = _ledger(tmp_path)
    bet = {"bet_id": "abc", "stake": 10, "price": 2.0, "label": "1", "market": "1X2",
           "selection": "home", "kickoff": "2026-01-01", "league": "E0",
           "home": "A", "away": "B", "edge": 0.05}
    assert led.add_bet(bet) is True
    assert led.add_bet(bet) is False
    assert len(led.bets) == 1
    assert led.bankroll == 1000.0  # la mise ne déduit pas la bankroll


def test_reglement_gain(tmp_path):
    led = _ledger(tmp_path)
    led.add_bet({"bet_id": "b1", "stake": 10, "price": 2.5, "label": "1", "market": "1X2",
                 "selection": "home", "kickoff": "2026-01-01", "league": "E0",
                 "home": "A", "away": "B", "edge": 0.05})
    assert led.settle_bet("b1", STATUS_WON, 15.0, result="2-1", closing_price=2.2)
    assert led.bankroll == pytest.approx(1015.0)
    b = led.bets[0]
    assert b["clv"] == pytest.approx(2.5 / 2.2 - 1, abs=1e-4)
    stats = led.stats()
    assert stats["n_settled"] == 1 and stats["roi"] == pytest.approx(1.5)


def test_reglement_perte_et_stats(tmp_path):
    led = _ledger(tmp_path)
    led.add_bet({"bet_id": "b1", "stake": 20, "price": 2.0, "label": "1", "market": "1X2",
                 "selection": "home", "kickoff": "2026-01-01", "league": "E0",
                 "home": "A", "away": "B", "edge": 0.05})
    led.settle_bet("b1", STATUS_LOST, -20.0, result="0-1")
    stats = led.stats()
    assert stats["bankroll"] == 980.0
    assert stats["roi"] == pytest.approx(-1.0)
    assert stats["n_open"] == 0


def test_pas_de_double_reglement(tmp_path):
    led = _ledger(tmp_path)
    led.add_bet({"bet_id": "b1", "stake": 10, "price": 2.0, "label": "1", "market": "1X2",
                 "selection": "home", "kickoff": "2026-01-01", "league": "E0",
                 "home": "A", "away": "B", "edge": 0.05})
    assert led.settle_bet("b1", STATUS_WON, 10.0)
    assert led.settle_bet("b1", STATUS_WON, 10.0) is False
    assert led.bankroll == pytest.approx(1010.0)
