"""Gemini (palier gratuit) : contexte rédactionnel et signaux de risque.

Utilisation strictement encadrée :
  - génère un commentaire court par match sélectionné (2-3 phrases)
  - signale les risques factuels connus (calendrier, enjeu, dynamique)
  - JAMAIS utilisé pour produire une probabilité : les modèles statistiques
    sont seuls décisionnaires. Le LLM n'écrit que du texte.

Si la clé est absente ou le quota épuisé, le système continue sans texte
(dégradation silencieuse, aucun impact sur les pronostics).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from ..http import HttpClient
from ..logging_conf import get_logger

log = get_logger("source.gemini")

BASE = "https://generativelanguage.googleapis.com/v1beta/models"


class GeminiClient:
    def __init__(self, cfg, client: HttpClient | None = None):
        llm = cfg.llm if hasattr(cfg, "llm") else cfg
        self.key = os.getenv("GEMINI_API_KEY", "").strip() or os.getenv("GOOGLE_API_KEY", "").strip()
        self.model = llm.get("model", "gemini-2.5-flash")
        self.fallbacks = list(llm.get("fallback_models", []))
        self.max_calls = int(llm.get("max_calls_per_run", 8))
        self.max_tokens = int(llm.get("max_output_tokens", 400))
        self.temperature = float(llm.get("temperature", 0.4))
        self.enabled = bool(self.key) and bool(llm.get("enabled", True))
        self.timeout = float(llm.get("timeout_s", 12))
        self._dead_models: set[str] = set()
        self._failures = 0
        self._max_failures = int(llm.get("max_failures_per_run", 3))
        self.http = client or HttpClient()
        self.cache_dir = Path(cfg.root) / "data" / "cache" / "llm"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.calls = 0

    # ------------------------------------------------------------------ core
    def _cache_file(self, prompt: str) -> Path:
        h = hashlib.sha256((self.model + prompt).encode()).hexdigest()[:24]
        return self.cache_dir / f"{h}.json"

    def generate(self, prompt: str, system: str | None = None) -> str | None:
        if not self.enabled or self.calls >= self.max_calls:
            return None
        cached = self._cache_file(prompt)
        if cached.exists():
            try:
                return json.loads(cached.read_text(encoding="utf-8")).get("text")
            except json.JSONDecodeError:
                pass
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_tokens,
                "topP": 0.9,
            },
            "safetySettings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_LOW_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_LOW_AND_ABOVE"},
            ],
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        self.calls += 1
        for model in [self.model] + self.fallbacks:
            if model in self._dead_models:
                continue
            url = f"{BASE}/{model}:generateContent?key={self.key}"
            data = self._post(url, payload)
            text = self._extract(data)
            if text:
                cached.write_text(json.dumps({"text": text, "model": model}), encoding="utf-8")
                self._failures = 0
                return text
            self._failures += 1
            if self._failures >= self._max_failures:
                # inutile de continuer à attendre : le briefing est un bonus,
                # pas une dépendance du pipeline.
                log.warning("Gemini indisponible : désactivé pour cette exécution")
                self.enabled = False
                return None
        return None

    def _post(self, url: str, payload: dict):
        import requests
        try:
            r = requests.post(url, json=payload, timeout=self.timeout,
                              headers={"Content-Type": "application/json"})
            if r.status_code == 429:
                log.warning("Gemini: quota atteint (429)")
                return None
            if r.status_code >= 400:
                log.warning("Gemini HTTP %s sur %s", r.status_code, url.split("?")[0])
                if r.status_code in {400, 401, 403, 404}:
                    # modèle inexistant ou clé refusée : ne plus le réessayer
                    try:
                        model = url.split("/models/")[1].split(":")[0]
                        self._dead_models.add(model)
                    except IndexError:
                        pass
                return None
            return r.json()
        except Exception as exc:  # noqa: BLE001
            log.warning("Gemini: échec réseau (%s)", exc)
            return None

    @staticmethod
    def _extract(data) -> str | None:
        try:
            cands = data.get("candidates") or []
            if not cands:
                return None
            parts = cands[0].get("content", {}).get("parts", [])
            txt = " ".join(p.get("text", "") for p in parts).strip()
            return txt or None
        except Exception:  # noqa: BLE001
            return None

    # --------------------------------------------------------------- métier
    def match_briefing(self, league: str, home: str, away: str, kickoff: str,
                       probs: dict[str, float], xg: dict[str, float],
                       value_note: str = "") -> str | None:
        prompt = (
            f"Match de football : {home} - {away} ({league}), coup d'envoi {kickoff}.\n"
            f"Modèle statistique ATHENA : domicile {probs.get('home', 0):.1%}, "
            f"nul {probs.get('draw', 0):.1%}, extérieur {probs.get('away', 0):.1%}. "
            f"Buts attendus : {xg.get('home', 0):.2f} vs {xg.get('away', 0):.2f}.\n"
            f"{value_note}\n"
            "Écris en français un briefing de 2 phrases maximum (45 mots max) : "
            "l'enjeu du match et le point tactique clé. Pas de conseil de pari, "
            "pas de cote, pas d'invention de blessure : uniquement du contexte football."
        )
        return self.generate(prompt, system=(
            "Tu es un analyste football francophone, factuel et concis. "
            "Tu n'inventes jamais de faits précis (blessures, compositions) que tu ne connais pas."
        ))
