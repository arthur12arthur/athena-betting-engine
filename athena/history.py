"""Historique des pronostics : archive, règlement a posteriori, statistiques.

Chaque exécution enregistre les pronostics publiés. Quand le match est joué,
les lignes sont complétées avec le score réel, l'issue et le résultat
(juste/faux). On obtient ainsi un historique vérifiable, distinct du ledger
des paris : il mesure la **qualité de prédiction**, pas la chance aux paris.

Fichier : ``data/state/predictions_history.json`` (versionné, diffable).
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .logging_conf import get_logger

log = get_logger("history")

OUTCOMES = ("home", "draw", "away")
MAX_ENTRIES = 6000


def pred_id(league: str, kickoff: str, home: str, away: str) -> str:
    raw = f"{league}|{str(kickoff)[:10]}|{home}|{away}"
    return hashlib.sha1(raw.encode()).hexdigest()[:14]


class PredictionArchive:
    def __init__(self, path: Path, entries: list[dict] | None = None):
        self.path = path
        self.entries: list[dict] = entries or []
        self.by_id = {e["pred_id"]: e for e in self.entries}

    # ------------------------------------------------------------------ I/O
    @classmethod
    def load(cls, path: Path) -> PredictionArchive:
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                entries = data.get("predictions", data if isinstance(data, list) else [])
                return cls(path, entries)
            except json.JSONDecodeError:
                log.warning("archive de pronostics illisible, réinitialisation")
        return cls(path, [])

    def save(self) -> None:
        self.entries = self.entries[-MAX_ENTRIES:]
        self.by_id = {e["pred_id"]: e for e in self.entries}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump({"updated_at": datetime.now(UTC).isoformat(timespec="seconds"),
                       "predictions": self.entries}, fh, ensure_ascii=False)
        os.replace(tmp, self.path)

    # ------------------------------------------------------------- mutations
    def add_many(self, preds) -> int:
        added = 0
        for p in preds:
            pid = pred_id(p["league"], p["kickoff"], p["home"], p["away"])
            if pid in self.by_id:
                continue
            entry = {
                "pred_id": pid, "league": p["league"], "kickoff": str(p["kickoff"])[:10],
                "home": p["home"], "away": p["away"],
                "p": {"home": round(float(p["p_home"]), 4),
                      "draw": round(float(p["p_draw"]), 4),
                      "away": round(float(p["p_away"]), 4)},
                "xg": {"home": round(float(p["lam_home"]), 2),
                       "away": round(float(p["lam_away"]), 2)},
                "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "result": None, "outcome": None, "hit": None,
            }
            self.entries.append(entry)
            self.by_id[pid] = entry
            added += 1
        return added

    def settle(self, results: dict[tuple, tuple[int, int]]) -> int:
        """Complète les pronostics dont le résultat est connu.

        ``results`` : {(league, 'YYYY-MM-DD', home, away): (buts_dom, buts_ext)}
        """
        n = 0
        for e in self.entries:
            if e.get("result") is not None:
                continue
            key = (e["league"], e["kickoff"], e["home"], e["away"])
            score = results.get(key)
            if score is None:
                for k, v in results.items():
                    if (k[0] == e["league"] and k[2] == e["home"] and k[3] == e["away"]
                            and abs((datetime.fromisoformat(k[1]).date()
                                     - datetime.fromisoformat(e["kickoff"]).date()).days) <= 2):
                        score = v
                        break
            if score is None:
                continue
            hg, ag = score
            outcome = "home" if hg > ag else ("draw" if hg == ag else "away")
            e["result"] = f"{hg}-{ag}"
            e["outcome"] = outcome
            e["hit"] = bool(max(e["p"], key=e["p"].get) == outcome)
            e["settled_at"] = datetime.now(UTC).isoformat(timespec="seconds")
            n += 1
        return n

    # -------------------------------------------------------------- analytics
    def settled(self) -> list[dict]:
        return [e for e in self.entries if e.get("result") is not None]

    def stats(self, last_days: int | None = None) -> dict:
        rows = self.settled()
        if last_days:
            cutoff = (datetime.now(UTC).date()).toordinal() - last_days
            rows = [e for e in rows
                    if datetime.fromisoformat(e["kickoff"]).date().toordinal() >= cutoff]
        if not rows:
            return {"n": 0, "hit_rate": None, "logloss": None, "brier": None,
                    "n_total": len(self.entries), "by_league": []}
        P = np.array([[e["p"]["home"], e["p"]["draw"], e["p"]["away"]] for e in rows])
        y = np.array([OUTCOMES.index(e["outcome"]) for e in rows])
        P = np.clip(P, 1e-9, 1.0)
        ll = float(-np.log(P[np.arange(len(y)), y]).mean())
        brier = float(((P - np.eye(3)[y]) ** 2).sum(axis=1).mean())
        by_league: dict[str, list] = {}
        for e in rows:
            by_league.setdefault(e["league"], []).append(int(e["hit"]))
        return {
            "n": len(rows),
            "hit_rate": round(float(np.mean([int(e["hit"]) for e in rows])), 4),
            "logloss": round(ll, 5),
            "brier": round(brier, 5),
            "n_total": len(self.entries),
            "by_league": [
                {"league": k, "n": len(v), "hit_rate": round(float(np.mean(v)), 4)}
                for k, v in sorted(by_league.items(), key=lambda kv: -len(kv[1]))
            ],
        }

    def recent(self, n: int = 60) -> list[dict]:
        rows = [e for e in self.entries if e.get("result") is not None]
        return list(reversed(rows[-n:]))
