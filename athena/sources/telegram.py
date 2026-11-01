"""Notifications Telegram (Bot API officielle, gratuite).

Deux variables d'environnement suffisent :
  TELEGRAM_BOT_TOKEN  — jeton du bot (créé via @BotFather, gratuit)
  TELEGRAM_CHAT_ID    — id du chat/canal destinataire

Sans ces variables, le système fonctionne normalement : l'envoi est
simplement ignoré (aucune erreur, aucun blocage du pipeline).
"""
from __future__ import annotations

import html
import os
from datetime import UTC, datetime

import requests

from ..logging_conf import get_logger

log = get_logger("source.telegram")

API = "https://api.telegram.org/bot{token}/sendMessage"
MAX_LEN = 3900  # marge sous la limite Telegram de 4096 caractères


def _esc(v) -> str:
    return html.escape(str(v), quote=False)


class TelegramNotifier:
    def __init__(self, token: str | None = None, chat_id: str | None = None,
                 site_url: str | None = None, enabled: bool = True):
        self.token = (token or os.getenv("TELEGRAM_BOT_TOKEN", "")).strip()
        self.chat_id = (chat_id or os.getenv("TELEGRAM_CHAT_ID", "")).strip()
        self.site_url = site_url or os.getenv("ATHENA_SITE_URL", "").strip()
        self.enabled = enabled and bool(self.token) and bool(self.chat_id)
        self.sent = 0

    @classmethod
    def from_config(cls, cfg) -> TelegramNotifier:
        tg = (cfg.raw.get("telegram") or {})
        return cls(chat_id=tg.get("chat_id"), site_url=tg.get("site_url") or
                   os.getenv("ATHENA_SITE_URL", ""),
                   enabled=bool(tg.get("enabled", True)))

    # ------------------------------------------------------------------ envoi
    def send(self, text: str, silent: bool = False) -> bool:
        if not self.enabled:
            log.info("Telegram désactivé (token/chat_id absents) — message non envoyé")
            return False
        for chunk in self._split(text):
            try:
                r = requests.post(
                    API.format(token=self.token),
                    json={"chat_id": self.chat_id, "text": chunk, "parse_mode": "HTML",
                          "disable_web_page_preview": True,
                          "disable_notification": silent},
                    timeout=20,
                )
                if r.status_code == 429:
                    log.warning("Telegram: limite de débit atteinte")
                    return False
                if r.status_code >= 400:
                    log.warning("Telegram HTTP %s : %s", r.status_code, r.text[:180])
                    return False
                self.sent += 1
            except Exception as exc:  # noqa: BLE001
                log.warning("Telegram: échec réseau (%s)", exc)
                return False
        log.info("Telegram: message envoyé (%d bloc(s))", self.sent)
        return True

    @staticmethod
    def _split(text: str) -> list[str]:
        if len(text) <= MAX_LEN:
            return [text]
        out, buf = [], ""
        for line in text.split("\n"):
            if len(buf) + len(line) + 1 > MAX_LEN:
                out.append(buf)
                buf = ""
            buf += line + "\n"
        if buf:
            out.append(buf)
        return out

    # ------------------------------------------------------------- formatage
    def daily_message(self, *, fixtures: int, bets: list[dict], settled: list[dict],
                      stats: dict, model: dict, top: list[dict] | None = None,
                      n_pred_archive: int = 0, hit_rate: float | None = None) -> str:
        now = datetime.now(UTC).strftime("%d/%m/%Y %H:%M UTC")
        roi = stats.get("roi", 0.0) or 0.0
        clv = stats.get("clv_avg")
        lines = [
            "<b>ATHENA — rapport de pronostics</b>",
            f"<i>{now}</i>",
            "",
            f"🏦 Bankroll : <b>{stats.get('bankroll', 0):.2f} u</b> "
            f"({(stats.get('pnl_total', 0) or 0):+.2f} u · ROI {roi*100:+.1f} %)",
            f"📊 Paris : {stats.get('n_settled', 0)} réglés · {stats.get('n_open', 0)} en cours",
            f"🎯 Matchs analysés : {fixtures}",
        ]
        if clv is not None:
            lines.append(f"📈 CLV moyen : <b>{clv*100:+.2f} %</b>")
        if hit_rate is not None and n_pred_archive:
            lines.append(f"✅ Pronostics résolus : {n_pred_archive} · réussite {hit_rate*100:.1f} %")

        if settled:
            lines += ["", "<b>Résultats réglés</b>"]
            for b in settled[:8]:
                icon = {"won": "✅", "lost": "❌", "push": "↩️",
                        "half_won": "🟢", "half_lost": "🟠"}.get(b.get("status"), "•")
                lines.append(
                    f"{icon} {_esc(b.get('home'))}–{_esc(b.get('away'))} "
                    f"<b>{_esc(b.get('result', ''))}</b> · {_esc(b.get('label'))} "
                    f"@ {b.get('price', 0):.2f} → <b>{b.get('pnl', 0):+.2f} u</b>")

        if bets:
            lines += ["", "<b>Value bets du jour</b>"]
            for b in bets[:10]:
                lines.append(
                    f"• {_esc(b.get('home'))}–{_esc(b.get('away'))} "
                    f"({_esc(b.get('kickoff'))})\n"
                    f"   {_esc(b.get('label'))} @ <b>{b.get('price', 0):.2f}</b> · "
                    f"edge {b.get('edge', 0)*100:+.1f} % · mise {b.get('stake', 0):.2f} u"
                    + ("" if b.get("multi_book") else " · ⚠️ source unique"))
        else:
            lines += ["", "Aucun value bet retenu : le marché était efficient "
                      "ou les garde-fous ont bloqué les candidats."]

        if top:
            lines += ["", "<b>Pronostics les plus tranchés</b>"]
            for p in top[:5]:
                probs = p["p"]
                best = max(probs, key=probs.get)
                frac = {"home": f"{_esc(p['home'])} gagne", "draw": "match nul",
                        "away": f"{_esc(p['away'])} gagne"}[best]
                lines.append(f"• {_esc(p['home'])}–{_esc(p['away'])} : {frac} "
                             f"({probs[best]*100:.0f} %) · xG {p['xg']['home']:.1f}–{p['xg']['away']:.1f}")

        lines += ["", f"<i>modèle {_esc(model.get('version', '—'))} · "
                  f"apprentissage {model.get('n_matches_train', '?')} matchs</i>"]
        if self.site_url:
            lines.append(f'<a href="{_esc(self.site_url)}">Tableau de bord complet</a>')
        lines.append("<i>Paper trading — aucune mise réelle. 18+</i>")
        return "\n".join(lines)
