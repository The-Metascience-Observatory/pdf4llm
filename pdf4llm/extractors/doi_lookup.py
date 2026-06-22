"""
DOI-based title lookup via multiple metadata APIs.

Used as a last-resort title enrichment step when local extraction fails.
Chain: Crossref → DataCite → Unpaywall → Semantic Scholar → OpenAlex

Results are cached in-process to avoid redundant requests during batch runs.
"""

import logging
import os
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# In-process cache: doi -> title (or None if every API failed)
_title_cache: dict[str, Optional[str]] = {}

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/117.0.0.0 Safari/537.36"
    )
}


def _get(url: str, headers: dict, timeout: int = 8) -> Optional[requests.Response]:
    """GET with one retry on 5xx; returns None on timeout/connection errors."""
    for attempt in range(2):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            if r.status_code == 429 or r.status_code >= 500:
                wait = min(int(r.headers.get("Retry-After", 5 * (2 ** attempt))), 15)
                logger.debug(f"HTTP {r.status_code} from {url}, waiting {wait}s")
                time.sleep(wait)
                continue
            return r
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            return None
        except requests.exceptions.RequestException as e:
            logger.debug(f"Request error for {url}: {e}")
            if attempt == 0:
                time.sleep(1)
    return None


def _try_crossref(doi: str) -> Optional[str]:
    """Crossref Works API — most reliable for journal articles."""
    email = os.getenv("CROSSREFEMAIL") or os.getenv("CONTACT_EMAIL")
    url = f"https://api.crossref.org/works/{doi}"
    if email:
        url += f"?mailto={email}"

    headers = _HEADERS.copy()
    api_key = os.getenv("CROSSREF_API_KEY")
    if api_key:
        headers["Crossref-Plus-API-Token"] = f"Bearer {api_key}"

    r = _get(url, headers)
    if r and r.status_code == 200:
        try:
            titles = r.json().get("message", {}).get("title", [])
            if titles and titles[0]:
                return titles[0].strip()
        except Exception as e:
            logger.debug(f"_try_crossref parse error for {doi}: {e}")
    return None


def _try_datacite(doi: str) -> Optional[str]:
    """DataCite API — good coverage for preprints and datasets."""
    r = _get(f"https://api.datacite.org/dois/{doi.lower()}", _HEADERS)
    if r and r.status_code == 200:
        try:
            titles = r.json().get("data", {}).get("attributes", {}).get("titles", [])
            if titles and titles[0].get("title"):
                return titles[0]["title"].strip()
        except Exception as e:
            logger.debug(f"_try_datacite parse error for {doi}: {e}")
    return None


def _try_unpaywall(doi: str) -> Optional[str]:
    """Unpaywall — good journal coverage, no key needed (email optional)."""
    email = os.getenv("CROSSREFEMAIL") or os.getenv("CONTACT_EMAIL") or "noreply@example.com"
    r = _get(f"https://api.unpaywall.org/v2/{doi}?email={email}", _HEADERS)
    if r and r.status_code == 200:
        try:
            title = r.json().get("title")
            if title:
                return title.strip()
        except Exception as e:
            logger.debug(f"_try_unpaywall parse error for {doi}: {e}")
    return None


def _try_semantic_scholar(doi: str) -> Optional[str]:
    """Semantic Scholar Graph API — broad academic coverage, no key needed."""
    headers = _HEADERS.copy()
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key

    r = _get(
        f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}?fields=title",
        headers,
    )
    if r and r.status_code == 200:
        try:
            title = r.json().get("title")
            if title:
                return title.strip()
        except Exception as e:
            logger.debug(f"_try_semantic_scholar parse error for {doi}: {e}")
    return None


def _try_openalex(doi: str) -> Optional[str]:
    """OpenAlex — very broad coverage including grey literature."""
    email = os.getenv("CROSSREFEMAIL") or os.getenv("CONTACT_EMAIL")
    url = f"https://api.openalex.org/works/https://doi.org/{doi}"
    if email:
        url += f"?mailto={email}"
    api_key = os.getenv("OPENALEXAPIKEY")
    if api_key:
        sep = "&" if "?" in url else "?"
        url += f"{sep}api_key={api_key}"

    r = _get(url, _HEADERS)
    if r and r.status_code == 200:
        try:
            title = r.json().get("title")
            if title:
                return title.strip()
        except Exception as e:
            logger.debug(f"_try_openalex parse error for {doi}: {e}")
    return None


def fetch_title_from_doi(doi: str) -> Optional[str]:
    """
    Look up a paper title from its DOI.

    Tries APIs in order: Crossref → DataCite → Unpaywall →
    Semantic Scholar → OpenAlex. Stops as soon as one returns a title.

    Results are cached for the lifetime of the process. Returns None if
    all APIs fail or return no title.
    """
    if not doi:
        return None

    if doi in _title_cache:
        return _title_cache[doi]

    for fn in (_try_crossref, _try_datacite, _try_unpaywall,
               _try_semantic_scholar, _try_openalex):
        try:
            title = fn(doi)
        except Exception as e:
            logger.debug(f"{fn.__name__} raised for {doi}: {e}")
            title = None
        if title:
            logger.debug(f"Title for {doi} from {fn.__name__}: {title[:60]}")
            _title_cache[doi] = title
            return title

    _title_cache[doi] = None
    return None
