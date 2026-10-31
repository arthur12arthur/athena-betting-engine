"""Client HTTP commun : retries, backoff, cache disque, politesse."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import requests

from .logging_conf import get_logger

log = get_logger("http")

UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)


class HttpClient:
    """Wrapper requests avec retries + cache de réponse (idempotent)."""

    def __init__(self, cache_dir: Path | None = None, timeout: int = 25, retries: int = 3):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": UA, "Accept": "application/json,text/csv,*/*"})
        self.cache_dir = cache_dir
        self.timeout = timeout
        self.retries = retries
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ util
    def _cache_path(self, key: str, ext: str) -> Path | None:
        if not self.cache_dir:
            return None
        h = hashlib.sha256(key.encode()).hexdigest()[:20]
        return self.cache_dir / f"{h}.{ext}"

    def get(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        cache_hours: float = 0.0,
        ext: str = "json",
        raw: bool = False,
    ) -> Any:
        key = url + json.dumps(params or {}, sort_keys=True) + json.dumps(headers or {}, sort_keys=True)
        cpath = self._cache_path(key, ext)
        if cpath and cache_hours > 0 and cpath.exists():
            age_h = (time.time() - cpath.stat().st_mtime) / 3600
            if age_h <= cache_hours:
                data = cpath.read_text(encoding="utf-8", errors="replace")
                return data if raw else json.loads(data)

        last_err: Exception | None = None
        for attempt in range(self.retries):
            try:
                r = self.session.get(url, params=params, headers=headers, timeout=self.timeout)
                if r.status_code == 429:
                    wait = int(r.headers.get("Retry-After", 5))
                    log.warning("429 sur %s -> pause %ss", url, wait)
                    time.sleep(min(wait, 30))
                    continue
                if r.status_code >= 500:
                    raise requests.HTTPError(f"HTTP {r.status_code}")
                r.raise_for_status()
                text = r.text
                if cpath:
                    cpath.write_text(text, encoding="utf-8")
                if raw:
                    return text
                if not text.strip():
                    return None
                return json.loads(text)
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                if attempt < self.retries - 1:
                    time.sleep(1.5 * (attempt + 1))
        log.error("échec HTTP %s : %s", url, last_err)
        return None


# cache partagé (créé à la demande par les modules)
def default_client(cache_dir: Path | None = None, cache_hours: float = 6) -> HttpClient:
    return HttpClient(cache_dir=cache_dir)
