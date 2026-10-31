"""Normalisation et rapprochement des noms d'équipes.

Les trois sources utilisent des conventions différentes :
  - football-data.co.uk : noms abrégés      ("Man United", "Nott'm Forest")
  - ESPN                 : noms complets     ("Manchester United")
  - Pinnacle             : noms commerciaux  ("Manchester United", "Paris Saint Germain")

On canonise sur les noms football-data (référence de l'entraînement) via :
  1) normalisation (casse, accents, ponctuation, mots génériques)
  2) table d'alias versionnée (data/aliases.yaml)
  3) rapprochement flou (difflib) restreint aux équipes de la même ligue
"""
from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

import yaml

# mots génériques inutiles au rapprochement
STOPWORDS = {
    "fc", "afc", "cf", "sc", "ac", "as", "ss", "rc", "ud", "cd", "sv", "vfl", "vfb",
    "tsg", "bsc", "fsv", "spvgg", "club", "de", "del", "la", "le", "the", "calcio",
    "futebol", "football", "sad", "futbol", "fotbal", "kfc", "kv", "kaa", "rsc",
    "sk", "bk", "if", "ff", "fc.", "1.", "1899", "1846", "1900", "1904", "1907",
    "1913", "04", "05", "07", "09", "96", "98", "sco", "sa", "srl", "spa", "ssc",
    "us", "usc", "acf", "hellas", "verona", "sporting", "cp", "gd", "boavista",
}


def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def normalize(name: str) -> str:
    """Forme canonique comparable : minuscules, sans accent ni ponctuation."""
    if not name:
        return ""
    s = strip_accents(str(name)).lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    tokens = [t for t in s.split() if t and t not in STOPWORDS]
    return " ".join(tokens)


def slug(name: str) -> str:
    return normalize(name).replace(" ", "-")


def similarity(a: str, b: str) -> float:
    na, nb = normalize(a), normalize(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    base = SequenceMatcher(None, na, nb).ratio()
    # bonus si un nom est contenu dans l'autre ("inter" / "inter milan")
    ta, tb = set(na.split()), set(nb.split())
    if ta and tb and (ta <= tb or tb <= ta):
        base = max(base, 0.9)
    return base


class TeamResolver:
    """Résout un nom brut (ESPN/Pinnacle) vers le nom canonique football-data."""

    def __init__(self, aliases: dict[str, str] | None = None, teams_by_league: dict[str, list[str]] | None = None):
        self.aliases = {normalize(k): v for k, v in (aliases or {}).items()}
        self.teams_by_league = teams_by_league or {}
        self._index: dict[str, str] = {}
        for _league, teams in self.teams_by_league.items():
            for t in teams:
                self._index[normalize(t)] = t
        self._cache: dict[str, str | None] = {}

    # ------------------------------------------------------------------ load
    @classmethod
    def from_config(cls, cfg, teams_by_league: dict[str, list[str]] | None = None) -> TeamResolver:
        path = cfg.root / "data" / "aliases.yaml"
        aliases: dict[str, str] = {}
        if path.exists():
            with open(path, encoding="utf-8") as fh:
                aliases = yaml.safe_load(fh) or {}
        return cls(aliases=aliases, teams_by_league=teams_by_league)

    # --------------------------------------------------------------- resolve
    def _resolve_cached(self, raw: str, league: str | None, candidates_key: str) -> str | None:
        n = normalize(raw)
        if not n:
            return None
        if n in self.aliases:
            return self.aliases[n]
        if n in self._index:
            return self._index[n]
        pool = self.teams_by_league.get(league, []) if league else list(self._index.values())
        if not pool:
            pool = list(self._index.values())
        best, score = None, 0.0
        for cand in pool:
            s = similarity(raw, cand)
            if s > score:
                best, score = cand, s
        if score >= 0.84:
            return best
        return None

    def resolve(self, raw: str, league: str | None = None) -> str | None:
        mem_key = f"{league}|{normalize(raw)}"
        if mem_key in self._cache:
            return self._cache[mem_key]
        pool = self.teams_by_league.get(league, []) if league else []
        val = self._resolve_cached(raw, league, "|".join(pool))
        self._cache[mem_key] = val
        return val

    def resolve_or_self(self, raw: str, league: str | None = None) -> str:
        return self.resolve(raw, league) or raw

    def unresolved(self, raw: str, league: str | None = None) -> bool:
        return self.resolve(raw, league) is None
