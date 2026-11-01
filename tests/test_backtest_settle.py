import pytest

from athena.backtest.engine import settle
from athena.betting.settlement import settle_bool as _settle_market


@pytest.mark.parametrize("market,sel,line,hg,ag,expected", [
    ("1X2", "home", None, 2, 1, (True, False)),
    ("1X2", "draw", None, 1, 1, (True, False)),
    ("1X2", "away", None, 0, 3, (True, False)),
    ("1X2", "home", None, 1, 1, (False, False)),
    ("DC", "1X", None, 1, 1, (True, False)),
    ("DC", "12", None, 2, 0, (True, False)),
    ("DC", "X2", None, 0, 0, (True, False)),
    ("DC", "12", None, 1, 1, (False, False)),
    ("OU25", "over", 2.5, 2, 2, (True, False)),
    ("OU25", "under", 2.5, 1, 1, (True, False)),
    ("BTTS", "yes", None, 1, 1, (True, False)),
    ("BTTS", "no", None, 2, 0, (True, False)),
    ("AH", "home", -0.5, 2, 1, (True, False)),
    ("AH", "away", -0.5, 0, 1, (True, False)),
    ("AH", "home", -1.0, 2, 1, (False, True)),
    ("AH", "home", -1.0, 3, 1, (True, False)),
])
def test_reglement_tous_marches(market, sel, line, hg, ag, expected):
    assert settle(market, sel, line, hg, ag) == expected
    assert _settle_market(market, sel, line, hg, ag) == expected
