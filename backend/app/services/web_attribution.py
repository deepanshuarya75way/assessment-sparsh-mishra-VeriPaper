"""Web Provenance Attribution (WPA) via the Brave Search API.

Attributes text segments flagged as similar to live web sources with
URLs, titles, and per-source similarity percentages. Fully optional:
when BRAVE_API_KEY is absent or the API is rate-limited, the module
reports itself as unavailable and the rest of the pipeline continues.

Free tier: 2,000 queries/month. The caller should limit sampling to the
paper's most-suspicious segments (we suggest at most 5 per paper).
"""
import logging
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

import requests

logger = logging.getLogger(__name__)

BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
BRAVE_KEY_ENVS = ("BRAVE_API_KEY", "RENDER_GITHUB_TOKEN", "GITHUB_TOKEN")
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
    matches: List[WebMatch] = field(default_factory=list)
    segments_checked: int = 0
    summary: str = ""


def _get_brave_key() -> Optional[str]:
    for env in BRAVE_KEY_ENVS:
        key = (os.environ.get(env) or "").strip()
        if key and key.startswith(("BSA", "bsa")):
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


def attribute_web_sources(segments: List[str]) -> WebAttributionResult:
    key = _get_brave_key()
    if not key:
        return WebAttributionResult(
            available=False, summary="Web source check is not configured — add BRAVE_API_KEY to enable attribution.",
        )
    if not segments:
        return WebAttributionResult(available=True, summary="No text segments were flagged for web checks.")

    matches: List[WebMatch] = []
    checked = 0
    for segment in segments[:MAX_SEGMENTS]:
        checked += 1
        try:
            resp = requests.get(
                BRAVE_ENDPOINT,
                params={"q": segment[:200], "count": 5},
                headers={"Accept": "application/json", "Accept-Encoding": "gzip", "X-Subscription-Token": key},
                timeout=15,
            )
        except requests.RequestException as exc:  # noqa: BLE001
            logger.warning("Brave query failed: %s", exc)
            continue
        if resp.status_code == 429:
            return WebAttributionResult(
                available=False, matches=matches, segments_checked=checked,
                summary="Web source check temporarily unavailable (search quota reached). Corpus checks still apply.",
            )
        if resp.status_code != 200:
            logger.warning("Brave returned status %s", resp.status_code)
            continue
        data = resp.json()
        for item in (data.get("web", {}).get("results") or [])[:5]:
            score = _token_overlap(segment, f"{item.get('title', '')} {item.get('description', '')}")
            if score >= 25:
                matches.append(WebMatch(
                    url=item.get("url", ""),
                    title=item.get("title", ""),
                    snippet=(item.get("description") or "")[:240],
                    matched_text=segment[:140],
                    similarity=score,
                ))
    if matches:
        matches.sort(key=lambda m: m.similarity, reverse=True)
        summary = f"{len(matches)} live web source(s) resemble the paper's flagged text."
    else:
        summary = "No live web sources closely resemble the flagged text segments."
    return WebAttributionResult(available=True, matches=matches[:15], segments_checked=checked, summary=summary)
