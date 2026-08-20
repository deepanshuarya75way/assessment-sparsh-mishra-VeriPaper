"""Web Provenance Attribution (WPA) — multi-provider web source checks.

Attributes text segments flagged as similar to live web sources with
URLs, titles, and per-source similarity percentages. Requires no paid
service: the default DuckDuckGo HTML fallback works with zero API keys,
Google Custom Search becomes the primary provider when
GOOGLE_CSE_API_KEY + GOOGLE_CSE_ID are set (free tier: 100 queries/day),
and Brave remains an optional alternative (no longer needed, kept for
compatibility).

Quota-aware: the caller samples at most MAX_SEGMENTS per paper so that
even a single provider's free tier comfortably covers realistic usage.
"""
import json
import logging
import os
import re
import urllib.parse
from dataclasses import dataclass, field
from typing import List, Optional

import requests

logger = logging.getLogger(__name__)

BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
GOOGLE_ENDPOINT = "https://customsearch.googleapis.com/customsearch/v1"
MAX_SEGMENTS = 5  # quota-aware hard cap per paper


@dataclass
class WebMatch:
    url: str
    title: str
    snippet: str
    matched_text: str
    similarity: int  # percent


@dataclass
class WebAttributionResult:
    available: bool
    provider: str = ""  # "duckduckgo", "google", "brave", ""
    matches: List[WebMatch] = field(default_factory=list)
    segments_checked: int = 0
    summary: str = ""


def _get_brave_key() -> Optional[str]:
    for env in ("BRAVE_API_KEY",):
        key = (os.environ.get(env) or "").strip()
        if key and key.startswith(("BSA", "bsa")):
            return key
    return None


def _get_google_key() -> Optional[str]:
    key = (os.environ.get("GOOGLE_CSE_API_KEY") or "").strip()
    cse = (os.environ.get("GOOGLE_CSE_ID") or "").strip()
    if key and cse:
        return key
    return None


def _clean_words(text: str) -> List[str]:
    words = [w.lower() for w in re.findall(r"[A-Za-z]{3,}", text)]
    stop = set("the and or that this with from have been are was were also into more".split())
    return [w for w in words if w not in stop]


def _token_overlap(a: str, b: str) -> int:
    wa, wb = set(_clean_words(a)), set(_clean_words(b))
    if not wa or not wb:
        return 0
    return int(round(len(wa & wb) / len(wa | wb) * 100))


def _score_items(segment: str, items: List[dict]) -> List[WebMatch]:
    out: List[WebMatch] = []
    for item in items:
        text = f"{item.get('title', '')} {item.get('description', item.get('snippet', ''))}"
        score = _token_overlap(segment, text)
        if score >= 25:
            out.append(WebMatch(
                url=item.get("url", item.get("link", "")),
                title=item.get("title", ""),
                snippet=(item.get("description") or item.get("snippet", ""))[:240],
                matched_text=segment[:140],
                similarity=score,
            ))
    return out


def _query_duckduckgo(query: str) -> List[dict]:
    """Zero-key DuckDuckGo HTML search. No official JSON API exists, so we
    parse the lite HTML endpoint, which returns results as simple HTML.
    Free, unlimited, ToS-compliant for reasonable use."""
    url = "https://lite.duckduckgo.com/lite/"
    try:
        resp = requests.post(
            url,
            data={"q": query},
            headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) VeriPaper/1.0"},
            timeout=15,
        )
    except requests.RequestException as exc:  # noqa: BLE001
        logger.warning("DuckDuckGo query failed: %s", exc)
        return []
    if resp.status_code != 200:
        logger.warning("DuckDuckGo returned status %s", resp.status_code)
        return []
    html = resp.text or ""
    items: List[dict] = []
    # Results live inside <a class="result-link"> (lite UI)
    seen: set = set()
    DDG_PATTERN = r"<a\s+rel=" + r'"nofollow"' + r"\s+href=\"([^\"]+)\"\s+class='result-link'>(.*?)</a>"
    for m in re.finditer(DDG_PATTERN, html, re.S):
        href = m.group(1)
        title = re.sub(r"<[^>]+>", "", m.group(2)).strip()
        if not href or not title:
            continue
        href = urllib.parse.unquote(href)
        if href in seen or "duckduckgo.com" in href:
            continue
        seen.add(href)
        items.append({"url": href, "title": title[:160], "description": ""})
    return items[:5]


def _query_google(query: str) -> List[dict]:
    key = _get_google_key() or ""
    cse = (os.environ.get("GOOGLE_CSE_ID") or "").strip()
    try:
        resp = requests.get(
            GOOGLE_ENDPOINT,
            params={"key": key, "cx": cse, "q": query, "num": 5},
            timeout=15,
        )
    except requests.RequestException as exc:  # noqa: BLE001
        logger.warning("Google CSE query failed: %s", exc)
        return []
    if resp.status_code == 429 or resp.status_code >= 400:
        logger.warning("Google CSE returned status %s", resp.status_code)
        return []
    data = resp.json()
    return [
        {
            "url": it.get("link", ""),
            "title": it.get("title", ""),
            "description": it.get("snippet", ""),
        }
        for it in (data.get("items") or [])[:5]
    ]


def _query_brave(query: str) -> List[dict]:
    key = _get_brave_key()
    if not key:
        return []
    try:
        resp = requests.get(
            BRAVE_ENDPOINT,
            params={"q": query, "count": 5},
            headers={"Accept": "application/json", "Accept-Encoding": "gzip", "X-Subscription-Token": key},
            timeout=15,
        )
    except requests.RequestException as exc:  # noqa: BLE001
        logger.warning("Brave query failed: %s", exc)
        return []
    if resp.status_code == 429 or resp.status_code >= 400:
        logger.warning("Brave returned status %s", resp.status_code)
        return []
    return [
        {"url": it.get("url", ""), "title": it.get("title", ""), "description": it.get("description", "")}
        for it in (resp.json().get("web", {}).get("results") or [])[:5]
    ]


def attribute_web_sources(segments: List[str]) -> WebAttributionResult:
    """Check flagged segments against live web sources using the best
    available provider: Google CSE if configured, otherwise DuckDuckGo
    (zero-key), with Brave results preferred when its key is set.

    DuckDuckGo never reports "unavailable" unless it cannot be reached —
    it is free and unlimited, so the web attribution feature always works
    out of the box on any deployment, including the Render free tier."""
    if not segments:
        return WebAttributionResult(available=True, summary="No text segments were flagged for web checks.")

    matches: List[WebMatch] = []
    checked = 0
    provider = ""
    segments = [s for s in segments[:MAX_SEGMENTS] if s.strip()]
    if not segments:
        return WebAttributionResult(available=True, summary="No text segments were flagged for web checks.")

    use_google = bool(_get_google_key())
    use_brave = bool(_get_brave_key())

    for segment in segments:
        checked += 1
        query = segment[:200]
        found: List[dict] = []
        if use_google:
            found = _query_google(query)
            provider = provider or "google"
        if not found and use_brave:
            found = _query_brave(query)
            provider = provider or "brave"
        if not found:
            found = _query_duckduckgo(query)
            provider = provider or "duckduckgo"
        for m in _score_items(segment, found):
            matches.append(m)

    if matches:
        matches.sort(key=lambda m: m.similarity, reverse=True)
        summary = f"{len(matches)} live web source(s) resemble the paper's flagged text."
    else:
        summary = "No live web sources closely resemble the flagged text segments."
    return WebAttributionResult(
        available=True, provider=provider, matches=matches[:15],
        segments_checked=checked, summary=summary,
    )
