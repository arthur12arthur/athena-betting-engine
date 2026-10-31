"""État persistant : bankroll, paris ouverts, historique.

Choix technique : JSON versionné dans le dépôt (lisible, diffable, gratuit)
plutôt qu'une base de données binaire. Chaque exécution est atomique
(écriture dans un fichier temporaire puis remplacement).
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from .logging_conf import get_logger

log = get_logger("state")

STATUS_OPEN = "open"
STATUS_WON = "won"
STATUS_LOST = "lost"
STATUS_PUSH = "push"
STATUS_VOID = "void"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Ledger:
    """Portefeuille de paris (paper trading) + courbe de capital."""

    def __init__(self, path: Path, data: dict):
        self.path = path
        self.data = data

    # ------------------------------------------------------------------ I/O
    @classmethod
    def load(cls, path: Path, initial_bankroll: float = 1000.0) -> Ledger:
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                return cls(path, data)
            except json.JSONDecodeError:
                log.warning("ledger corrompu, réinitialisation")
        data = {
            "created_at": _now(),
            "initial_bankroll": initial_bankroll,
            "bankroll": initial_bankroll,
            "bets": [],
            "equity": [{"ts": _now(), "bankroll": initial_bankroll}],
            "runs": [],
        }
        led = cls(path, data)
        led.save()
        return led

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(self.data, fh, ensure_ascii=False, indent=1)
        os.replace(tmp, self.path)

    # ------------------------------------------------------------- accesseurs
    @property
    def bankroll(self) -> float:
        return float(self.data.get("bankroll", 0.0))

    @property
    def initial(self) -> float:
        return float(self.data.get("initial_bankroll", 1000.0))

    @property
    def bets(self) -> list[dict]:
        return self.data.setdefault("bets", [])

    def open_bets(self) -> list[dict]:
        return [b for b in self.bets if b.get("status") == STATUS_OPEN]

    def settled_bets(self) -> list[dict]:
        return [b for b in self.bets if b.get("status") in {STATUS_WON, STATUS_LOST, STATUS_PUSH}]

    def has_bet(self, bet_id: str) -> bool:
        return any(b.get("bet_id") == bet_id for b in self.bets)

    # --------------------------------------------------------------- mutations
    def add_bet(self, bet: dict) -> bool:
        if self.has_bet(bet["bet_id"]):
            return False
        bet.setdefault("status", STATUS_OPEN)
        bet.setdefault("created_at", _now())
        bet.setdefault("pnl", 0.0)
        self.bets.append(bet)
        # la bankroll représente le capital total (cash + mises engagées) :
        # elle ne bouge qu'au règlement, l'exposition est suivie séparément.
        return True

    def settle_bet(self, bet_id: str, status: str, pnl: float,
                   result: str | None = None, closing_price: float | None = None) -> bool:
        for b in self.bets:
            if b.get("bet_id") == bet_id and b.get("status") == STATUS_OPEN:
                b["status"] = status
                b["pnl"] = round(float(pnl), 2)
                b["settled_at"] = _now()
                b["bankroll_after"] = round(self.bankroll + float(pnl), 2)
                if result:
                    b["result"] = result
                if closing_price:
                    b["closing_price"] = closing_price
                    b["clv"] = round(float(b["price"]) / float(closing_price) - 1.0, 5)
                self.data["bankroll"] = round(self.bankroll + float(pnl), 2)
                self._mark_equity()
                return True
        return False

    def _mark_equity(self) -> None:
        eq = self.data.setdefault("equity", [])
        eq.append({"ts": _now(), "bankroll": round(self.bankroll, 2)})
        self.data["equity"] = eq[-2000:]

    def log_run(self, info: dict) -> None:
        runs = self.data.setdefault("runs", [])
        runs.append({"ts": _now(), **info})
        self.data["runs"] = runs[-120:]

    # ------------------------------------------------------------------ stats
    def stats(self) -> dict:
        settled = self.settled_bets()
        staked = sum(float(b["stake"]) for b in settled)
        pnl = sum(float(b.get("pnl", 0.0)) for b in settled)
        wins = sum(1 for b in settled if b["status"] == STATUS_WON)
        clv = [float(b["clv"]) for b in settled if b.get("clv") is not None]
        eq = [float(x["bankroll"]) for x in self.data.get("equity", [])]
        peak, dd = -1e18, 0.0
        for v in eq:
            peak = max(peak, v)
            if peak > 0:
                dd = max(dd, (peak - v) / peak)
        return {
            "bankroll": round(self.bankroll, 2),
            "initial_bankroll": self.initial,
            "pnl_total": round(self.bankroll - self.initial, 2),
            "n_bets": len(self.bets),
            "n_open": len(self.open_bets()),
            "n_settled": len(settled),
            "n_won": wins,
            "win_rate": round(wins / len(settled), 4) if settled else 0.0,
            "staked": round(staked, 2),
            "roi": round(pnl / staked, 4) if staked else 0.0,
            "pnl_realized": round(pnl, 2),
            "avg_odds": round(sum(float(b["price"]) for b in settled) / len(settled), 3) if settled else 0.0,
            "avg_edge": round(sum(float(b.get("edge", 0)) for b in settled) / len(settled), 4) if settled else 0.0,
            "clv_avg": round(sum(clv) / len(clv), 4) if clv else None,
            "max_drawdown": round(dd, 4),
            "exposure": round(sum(float(b["stake"]) for b in self.open_bets()), 2),
        }
