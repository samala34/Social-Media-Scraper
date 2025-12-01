# File: scrapingbee_client.py  (suggested replacement for scraper.py)
"""
scrapingbee_client.py
---------------------
Small, testable ScrapingBee client encapsulated in a class. Exposes a simple
fetch_html(...) function for compatibility with existing code that imports
fetch_html from scraper.py.
"""

import os
import logging
from typing import Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ENDPOINT = "https://app.scrapingbee.com/api/v1/"

class ScrapingBeeClient:
    """Simple client for ScrapingBee API with retries and optional per-call API key."""

    def __init__(self, api_key: Optional[str] = None, retries: int = 2, backoff: float = 0.3):
        """
        api_key: optional default key (still can be overridden per call)
        retries/backoff: settings for transient-network retry behaviour
        """
        self.default_key = (api_key or os.getenv("SCRAPINGBEE_KEY") or "").strip() or None
        self.session = self._session_with_retries(retries, backoff)

    @staticmethod
    def _session_with_retries(total_retries=2, backoff_factor=0.3):
        s = requests.Session()
        retries = Retry(
            total=total_retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=frozenset(["GET", "POST"])
        )
        s.mount("https://", HTTPAdapter(max_retries=retries))
        return s

    def _resolve_key(self, api_key: Optional[str]) -> Optional[str]:
        if api_key and str(api_key).strip():
            return str(api_key).strip()
        return self.default_key

    def validate_key(self, api_key: Optional[str] = None) -> (bool, str):
        """Quickly validate a key by requesting google.com (safe, no JS required)."""
        key = self._resolve_key(api_key)
        if not key:
            return False, "No key provided."
        try:
            r = requests.get(ENDPOINT, params={"api_key": key, "url": "https://www.google.com"}, timeout=10)
            if r.status_code == 200:
                return True, "OK"
            return False, f"Status {r.status_code}: {r.text[:400]}"
        except Exception as e:
            return False, f"Network error: {e}"

    def fetch_html(self,
                   url: str,
                   render_js: bool = True,
                   save_path: Optional[str] = None,
                   timeout: int = 60,
                   api_key: Optional[str] = None) -> str:
        """
        Fetch HTML (renders JS when requested). Raises RuntimeError on failure.
        api_key overrides configured/default key for this call.
        """
        key = self._resolve_key(api_key)
        if not key:
            raise RuntimeError("No ScrapingBee API key found. Pass api_key or set SCRAPINGBEE_KEY env var.")

        params = {"api_key": key, "url": url, "render_js": "true" if render_js else "false"}
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        try:
            r = self.session.get(ENDPOINT, params=params, headers=headers, timeout=timeout)
        except requests.RequestException as e:
            logger.exception("Network/request error when calling ScrapingBee: %s", e)
            raise RuntimeError(f"Network error calling ScrapingBee: {e}") from e

        if r.status_code == 200:
            html = r.text
            if save_path:
                try:
                    import os
                    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
                    with open(save_path, "w", encoding="utf-8") as f:
                        f.write(html)
                except Exception:
                    logger.exception("Failed to save fetched HTML to %s", save_path)
            return html

        body = r.text or ""
        if r.status_code == 401:
            logger.error("ScrapingBee 401 Unauthorized. Provider message: %s", body[:1000])
            raise RuntimeError(f"ScrapingBee 401 Unauthorized. Provider message: {body[:500]}")
        logger.error("ScrapingBee request failed: status=%s body=%s", r.status_code, body[:1000])
        raise RuntimeError(f"ScrapingBee request failed: status {r.status_code}, message: {body[:500]}")


# Backwards-compatible wrapper function named fetch_html so existing imports (from scraper import fetch_html)
# continue to work. If you prefer to keep the old filename, you can place this in scraper.py and import client.
_default_client = ScrapingBeeClient()

def fetch_html(url: str, render_js: bool = True, save_path: Optional[str] = None, timeout: int = 60, api_key: Optional[str] = None) -> str:
    """
    Compatibility wrapper: same signature as used previously (we add api_key optional param).
    Existing code that calls fetch_html(url, render_js=..., ...) will continue to work.
    """
    return _default_client.fetch_html(url, render_js=render_js, save_path=save_path, timeout=timeout, api_key=api_key)
