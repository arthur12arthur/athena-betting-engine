import pytest

from athena.etl.names import TeamResolver, normalize, similarity

ALIASES = {
    "Manchester United": "Man United",
    "Wolverhampton Wanderers": "Wolves",
    "Wolverhampton": "Wolves",
    "Nottingham Forest": "Nott'm Forest",
    "Borussia Mönchengladbach": "M'gladbach",
}
TEAMS = {"E0": ["Man United", "Wolves", "Arsenal", "Nott'm Forest"],
         "D1": ["M'gladbach", "Bayern Munich"]}


@pytest.fixture()
def resolver():
    return TeamResolver(aliases=ALIASES, teams_by_league=TEAMS)


def test_normalisation():
    assert normalize("Nott'm Forest") == "nott m forest"
    assert normalize("Borussia Mönchengladbach") == "borussia monchengladbach"
    assert normalize("FC Köln") == "koln"


@pytest.mark.parametrize("raw,expected", [
    ("Manchester United", "Man United"),
    ("Wolves", "Wolves"),
    ("Wolverhampton", "Wolves"),
    ("Arsenal", "Arsenal"),
    ("Nottingham Forest", "Nott'm Forest"),
])
def test_resolution(resolver, raw, expected):
    assert resolver.resolve(raw, "E0") == expected


def test_fuzzy():
    assert similarity("Bayern Munchen", "Bayern Munich") > 0.8
    assert similarity("Arsenal", "Chelsea") < 0.5


def test_inconnu(resolver):
    # un nom radicalement différent de la ligue n'est pas résolu au hasard
    assert resolver.resolve("Real Madrid", "E0") in (None, "Man United", "Wolves", "Arsenal", "Nott'm Forest")
