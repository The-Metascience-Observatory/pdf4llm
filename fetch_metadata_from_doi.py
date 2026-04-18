import requests
import time
import logging
import os
import re
import json
import urllib.parse
from pathlib import Path
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

# Load OpenAlex API key from environment
def _get_openalex_api_key():
    """Load OpenAlex API key from .env.local or environment"""
    api_key = os.getenv('OPENALEXAPIKEY')
    if not api_key:
        # Try to load from .env.local in parent directory
        try:
            env_path = Path(__file__).parent.parent / '.env.local'
            if env_path.exists():
                with open(env_path) as f:
                    for line in f:
                        if line.startswith('OPENALEXAPIKEY='):
                            api_key = line.strip().split('=', 1)[1]
                            break
        except Exception:
            pass
    return api_key

OPENALEX_API_KEY = _get_openalex_api_key()


def _get_env_key(key_name):
    """Load an API key from environment or .env.local"""
    val = os.getenv(key_name)
    if not val:
        try:
            env_path = Path(__file__).parent.parent / '.env.local'
            if env_path.exists():
                with open(env_path) as f:
                    for line in f:
                        if line.startswith(f'{key_name}='):
                            val = line.strip().split('=', 1)[1]
                            break
        except Exception:
            pass
    return val

CORE_API_KEY = _get_env_key('COREAPIKEY')
SCOPUS_API_KEY = _get_env_key('SCOPUS_API_KEY')
DIMENSIONS_API_KEY = _get_env_key('DIMENSIONS_API_KEY')
SEMANTIC_SCHOLAR_API_KEY = _get_env_key('SEMANTIC_SCHOLAR_API_KEY')
CROSSREF_API_KEY = _get_env_key('CROSSREF_API_KEY')
CROSSREF_EMAIL = _get_env_key('CROSSREFEMAIL')
ENTREZ_API_KEY = _get_env_key('ENTREZ_EUTILS_API_KEY')
CONTACT_EMAIL = _get_env_key('CONTACT_EMAIL') or 'your_email@example.com'


def _format_initial(name):
    """Add periods after single-letter initials in an author name.
    e.g., 'J Lukas' → 'J. Lukas', 'Jonathan W Schooler' → 'Jonathan W. Schooler'"""
    if not name:
        return name
    parts = name.split()
    formatted = []
    for part in parts:
        if len(part) == 1 and part.isalpha():
            formatted.append(part + '.')
        else:
            formatted.append(part)
    return ' '.join(formatted)


def _is_initial_token(token):
    """Check if a token is an initial or compound initials.
    Matches: 'J', 'J.', 'A.C.', 'J.L.', 'AC', etc."""
    stripped = token.replace('.', '')
    if not stripped:
        return True
    # All single uppercase letters (possibly with periods between)
    return all(c.isupper() for c in stripped) and len(stripped) <= 3


def _count_full_first_names(authors_str):
    """Count how many authors have full (non-abbreviated) first names.
    Returns (full_count, total_count)."""
    if not authors_str or authors_str in (None, "", "NaN"):
        return 0, 0
    names = str(authors_str).split(';')
    full = 0
    total = 0
    for name in names:
        name = name.strip()
        if not name:
            continue
        # Handle "Last, First" format
        if ',' in name:
            parts = name.split(',', 1)
            first = parts[1].strip() if len(parts) > 1 else ''
        else:
            parts = name.split()
            first = parts[0] if parts else ''
        first_token = first.split()[0] if first.split() else ''
        total += 1
        if first_token and not _is_initial_token(first_token):
            full += 1
    return full, total


def _authors_have_abbreviations(authors_str):
    """Check if the author string has abbreviated first names."""
    if not authors_str or authors_str in (None, "", "NaN"):
        return False
    full, total = _count_full_first_names(authors_str)
    if total == 0:
        return False
    return full < total


def _new_authors_are_better(current, new):
    """Check if new authors string has more full first names than current."""
    if not new or new in (None, "", "NaN"):
        return False
    if not current or current in (None, "", "NaN"):
        return True
    cur_full, cur_total = _count_full_first_names(current)
    new_full, new_total = _count_full_first_names(new)
    # Don't accept if we lost many authors (likely wrong paper)
    if new_total < cur_total * 0.7:
        return False
    return new_full > cur_full


def _request_with_retry(url, headers=None, timeout=10, max_retries=3):
    """Make an HTTP GET request with exponential backoff on transient failures.
    Uses Retry-After header when available, with longer waits for 429 rate limits.
    Timeouts and connection errors skip retries to avoid wasting time on unresponsive APIs."""
    for attempt in range(max_retries):
        try:
            r = requests.get(url, timeout=timeout, headers=headers)
            if r.status_code == 429 or r.status_code >= 500:
                # Use Retry-After header if present, otherwise exponential backoff
                retry_after = r.headers.get('Retry-After')
                if retry_after:
                    try:
                        wait = min(int(retry_after), 10)  # cap at 10s
                    except ValueError:
                        wait = 5 * (2 ** attempt)  # fallback
                elif r.status_code == 429:
                    wait = 5 * (2 ** attempt)  # 5s, 10s, 20s for rate limits
                else:
                    wait = 2 ** attempt  # 1s, 2s, 4s for server errors
                logger.warning(f"HTTP {r.status_code} from {url}, retrying in {wait}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
                continue
            return r
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            logger.warning(f"Connection/timeout error for {url}: {e}, skipping")
            return None
        except requests.exceptions.RequestException as e:
            wait = 2 ** attempt
            logger.warning(f"Request error for {url}: {e}, retrying in {wait}s (attempt {attempt+1}/{max_retries})")
            time.sleep(wait)
    return None


def fetch_metadata_from_doi(doi, email=None, delay=0.2, enable_base=False, enable_core=False):
    if email is None:
        email = CONTACT_EMAIL
    """
    Progressive multi-API metadata enrichment:
    DataCite → arXiv (10.48550/ DOIs only) → Crossref → Unpaywall → PubMed → BASE → Scopus → Dimensions.ai → Semantic Scholar → CORE → DBLP → OpenCitations → Europe PMC → OpenAlex → doi.org BibTeX
    Stops early if all fields are filled (authors must have full first names to count as complete).
    Europe PMC is last because it only returns abbreviated initials for author names.
    """
    if not isinstance(doi, str) or not doi.strip():
        return None

    doi = doi.strip()

    headers = {
        # Pure Chrome-on-Windows user-agent (spoof)
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/117.0.0.0 Safari/537.36"
        )
    }

    meta = {k: None for k in ["authors", "title", "journal", "volume", "issue", "pages", "year", "url"]}

    def enrich(current, new):
        """Fill missing fields in current dict with non-empty values from new dict.
        For authors: also replace if new has more full (non-abbreviated) first names."""
        if not new:
            return current
        for k, v in new.items():
            if v in [None, "", "NaN"]:
                continue
            if current.get(k) in [None, "", "NaN"]:
                current[k] = v
            elif k == "authors" and _new_authors_are_better(current[k], v):
                current[k] = v
        return current

    def is_complete(m):
        """Check if all metadata fields are filled.
        Authors with abbreviated first names don't count as complete."""
        for k in m:
            val = m.get(k)
            if val in [None, "", "NaN"]:
                return False
            if k == "authors" and _authors_have_abbreviations(val):
                return False
        return True

    # ---------- 1️⃣ DataCite ----------
    try:
        r = _request_with_retry(f"https://api.datacite.org/dois/{doi.lower()}", headers=headers)
        if r and r.status_code == 200:
            d = r.json().get("data", {}).get("attributes", {})
            authors = []
            for a in d.get("creators", []):
                name = a.get("name") or f"{a.get('givenName','')} {a.get('familyName','')}".strip()
                if name:
                    authors.append(_format_initial(name))
            # Use container title if available; publisher is not the journal
            container = d.get("container", {}) or {}
            journal = container.get("title") or None
            dc = {
                "authors": "; ".join(authors) or None,
                "title": (d.get("titles") or [{}])[0].get("title"),
                "journal": journal,
                "year": d.get("publicationYear"),
                "url": d.get("url") or f"https://doi.org/{doi}",
            }
            meta = enrich(meta, dc)
            if is_complete(meta):
                return meta
        elif r:
            logger.warning(f"DataCite returned HTTP {r.status_code} for DOI {doi}")
    except Exception as e:
        logger.warning(f"DataCite error for DOI {doi}: {e}")

    # ---------- 1️⃣b arXiv (for 10.48550/ DOIs only) ----------
    if doi.startswith("10.48550/") and "arXiv." in doi:
        try:
            import xml.etree.ElementTree as ET
            arxiv_id = doi.split("arXiv.", 1)[-1]
            r = _request_with_retry(
                f"https://export.arxiv.org/api/query?id_list={arxiv_id}",
                headers=headers,
            )
            if r and r.status_code == 200:
                ns = {'a': 'http://www.w3.org/2005/Atom'}
                root = ET.fromstring(r.text)
                entry = root.find('a:entry', ns)
                if entry is not None:
                    title_el = entry.find('a:title', ns)
                    published_el = entry.find('a:published', ns)
                    authors = "; ".join(
                        el.find('a:name', ns).text
                        for el in entry.findall('a:author', ns)
                        if el.find('a:name', ns) is not None
                    ) or None
                    ax = {
                        "authors": authors,
                        "title": title_el.text.strip() if title_el is not None else None,
                        "year": int(published_el.text[:4]) if published_el is not None else None,
                        "url": f"https://arxiv.org/abs/{arxiv_id}",
                    }
                    meta = enrich(meta, ax)
                    if is_complete(meta):
                        return meta
            elif r:
                logger.warning(f"arXiv API returned HTTP {r.status_code} for {arxiv_id}")
        except Exception as e:
            logger.warning(f"arXiv API error for DOI {doi}: {e}")

    # ---------- 2️⃣ Crossref ----------
    try:
        # Add Crossref Plus API key if available (provides higher rate limits)
        crossref_headers = headers.copy()
        crossref_url = f"https://api.crossref.org/works/{doi}"
        if CROSSREF_EMAIL:
            crossref_url += f"?mailto={CROSSREF_EMAIL}"
        if CROSSREF_API_KEY:
            crossref_headers['Crossref-Plus-API-Token'] = f'Bearer {CROSSREF_API_KEY}'
        r = _request_with_retry(crossref_url, headers=crossref_headers)
        if r and r.status_code == 200:
            m = r.json()["message"]
            authors = []
            for a in m.get("author", []):
                parts = []
                if "given" in a: parts.append(a["given"])
                if "family" in a: parts.append(a["family"])
                name = _format_initial(" ".join(parts).strip())
                if name:
                    authors.append(name)
            year = (
                m.get("published-print", {}).get("date-parts", [[None]])[0][0]
                or m.get("published-online", {}).get("date-parts", [[None]])[0][0]
            )
            cr = {
                "authors": "; ".join(authors) or None,
                "title": (m.get("title") or [None])[0],
                "journal": (m.get("container-title") or [None])[0],
                "volume": m.get("volume"),
                "issue": m.get("issue"),
                "pages": m.get("page"),
                "year": year,
                "url": f"https://doi.org/{doi}",
            }
            meta = enrich(meta, cr)
            if is_complete(meta):
                return meta
        elif r:
            logger.warning(f"Crossref returned HTTP {r.status_code} for DOI {doi}")
    except Exception as e:
        logger.warning(f"Crossref error for DOI {doi}: {e}")

    # ---------- 3️⃣ Unpaywall ----------
    try:
        r = _request_with_retry(f"https://api.unpaywall.org/v2/{doi}?email={email}", headers=headers)
        if r and r.status_code == 200:
            u = r.json()
            best_loc = u.get("best_oa_location") or {}
            authors = "; ".join(
                [_format_initial(f"{a.get('given','')} {a.get('family','')}".strip()) for a in u.get("z_authors", [])]
            ) or None
            up = {
                "authors": authors,
                "title": u.get("title"),
                "journal": u.get("journal_name"),
                "volume": u.get("journal_volume"),
                "issue": u.get("journal_issue"),
                "pages": u.get("journal_pages"),
                "year": u.get("year"),
                "url": best_loc.get("url") or u.get("doi_url") or f"https://doi.org/{doi}",
            }
            meta = enrich(meta, up)
            if is_complete(meta):
                return meta
        elif r:
            logger.warning(f"Unpaywall returned HTTP {r.status_code} for DOI {doi}")
    except Exception as e:
        logger.warning(f"Unpaywall error for DOI {doi}: {e}")

    # ---------- 4️⃣ PubMed ----------
    try:
        # Add API key if available (increases rate limit from 3/s to 10/s)
        api_key_param = f"&api_key={ENTREZ_API_KEY}" if ENTREZ_API_KEY else ""
        search_url = (
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            f"?db=pubmed&term={doi}[AID]&retmode=json&email={email}{api_key_param}"
        )
        r = _request_with_retry(search_url, headers=headers)
        if r and r.status_code == 200:
            pmids = r.json().get("esearchresult", {}).get("idlist", [])
            if pmids:
                time.sleep(delay)
                summary_url = (
                    f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
                    f"?db=pubmed&id={pmids[0]}&retmode=json&email={email}{api_key_param}"
                )
                r2 = _request_with_retry(summary_url, headers=headers)
                if r2 and r2.status_code == 200:
                    result = r2.json().get("result", {})
                    d = result.get(pmids[0], {})
                    if d and isinstance(d, dict):
                        authors = "; ".join(
                            _format_initial(a.get("name", ""))
                            for a in d.get("authors", [])
                            if a.get("authtype") == "Author"
                        ) or None
                        pub_year = None
                        pubdate = d.get("pubdate", "")
                        if pubdate and len(pubdate) >= 4 and pubdate[:4].isdigit():
                            pub_year = int(pubdate[:4])
                        pm = {
                            "authors": authors,
                            "title": d.get("title"),
                            "journal": d.get("fulljournalname"),
                            "volume": d.get("volume"),
                            "issue": d.get("issue"),
                            "pages": d.get("pages"),
                            "year": pub_year,
                            "url": f"https://doi.org/{doi}",
                        }
                        meta = enrich(meta, pm)
                        if is_complete(meta):
                            return meta
        elif r:
            logger.warning(f"PubMed returned HTTP {r.status_code} for DOI {doi}")
    except Exception as e:
        logger.warning(f"PubMed error for DOI {doi}: {e}")
    time.sleep(delay)

    # ---------- 7️⃣ BASE (Bielefeld Academic Search Engine) ----------
    # Disabled by default: title-based search is slow and frequently times out
    # Pass enable_base=True to re-enable
    if enable_base:
        try:
            search_title = meta.get("title") or ""
            if search_title:
                q = urllib.parse.quote(search_title)
                r = _request_with_retry(
                    f"https://api.base-search.net/cgi-bin/BaseHttpSearchInterface.fcgi"
                    f"?func=PerformSearch&query=dctitle:{q}&format=json&hits=5",
                    headers=headers,
                )
                if r and r.status_code == 200:
                    results = r.json().get("response", {}).get("docs", [])
                    for doc in results[:5]:
                        fetched_title = doc.get("dctitle", "")
                        if isinstance(fetched_title, list):
                            fetched_title = fetched_title[0] if fetched_title else ""
                        ratio = SequenceMatcher(None, search_title.lower(), fetched_title.lower()).ratio()
                        if ratio >= 0.9:
                            authors_raw = doc.get("dcCreator") or doc.get("dccreator") or []
                            if isinstance(authors_raw, str):
                                authors_raw = [authors_raw]
                            authors = "; ".join(_format_initial(a) for a in authors_raw) or None
                            base_year = None
                            dcdate = doc.get("dcyear") or doc.get("dcdate", "")
                            if isinstance(dcdate, list):
                                dcdate = dcdate[0] if dcdate else ""
                            dcdate = str(dcdate)
                            if dcdate and len(dcdate) >= 4 and dcdate[:4].isdigit():
                                base_year = int(dcdate[:4])
                            ba = {
                                "authors": authors,
                                "title": fetched_title,
                                "journal": doc.get("dcsource") or doc.get("dcpublisher"),
                                "year": base_year,
                                "url": doc.get("dclink") or doc.get("dcidentifier"),
                            }
                            meta = enrich(meta, ba)
                            break
                    if is_complete(meta):
                        return meta
                elif r:
                    logger.warning(f"BASE returned HTTP {r.status_code}")
        except Exception as e:
            logger.warning(f"BASE error: {e}")

    # ---------- 8️⃣ Scopus / Elsevier ----------
    if SCOPUS_API_KEY:
        try:
            scopus_headers = {**headers, "X-ELS-APIKey": SCOPUS_API_KEY, "Accept": "application/json"}
            r = _request_with_retry(
                f"https://api.elsevier.com/content/abstract/doi/{doi}",
                headers=scopus_headers,
            )
            if r and r.status_code == 200:
                data = r.json()
                resp = data.get("abstracts-retrieval-response", {})
                coredata = resp.get("coredata", {})
                # Authors
                authors_obj = resp.get("authors", {}).get("author", [])
                if isinstance(authors_obj, dict):
                    authors_obj = [authors_obj]
                authors = "; ".join(
                    _format_initial(
                        f"{a.get('ce:given-name', '')} {a.get('ce:surname', '')}".strip()
                    )
                    for a in authors_obj
                    if a.get('ce:surname')
                ) or None
                # Year from coverDate (YYYY-MM-DD)
                cover_date = coredata.get("prism:coverDate", "")
                year_val = int(cover_date[:4]) if cover_date and len(cover_date) >= 4 and cover_date[:4].isdigit() else None
                sc = {
                    "authors": authors,
                    "title": coredata.get("dc:title"),
                    "journal": coredata.get("prism:publicationName"),
                    "volume": coredata.get("prism:volume"),
                    "issue": coredata.get("prism:issueIdentifier"),
                    "pages": coredata.get("prism:pageRange"),
                    "year": year_val,
                    "url": coredata.get("prism:url") or f"https://doi.org/{doi}",
                }
                meta = enrich(meta, sc)
                if is_complete(meta):
                    return meta
            elif r:
                logger.warning(f"Scopus returned HTTP {r.status_code} for DOI {doi}")
        except Exception as e:
            logger.warning(f"Scopus error for DOI {doi}: {e}")

    # ---------- 🔟 Dimensions.ai ----------
    if DIMENSIONS_API_KEY:
        try:
            import json as json_lib
            dim_headers = {**headers, "Authorization": f"Bearer {DIMENSIONS_API_KEY}"}
            query = f'search publications where doi="{doi}" return publications[doi+title+authors+journal+year+volume+issue+pages]'
            r = requests.post(
                "https://app.dimensions.ai/api/dsl.json",
                headers=dim_headers,
                json={"query": query},
                timeout=10
            )
            if r and r.status_code == 200:
                data = r.json()
                pubs = data.get("publications", [])
                if pubs:
                    p = pubs[0]
                    authors_list = p.get("authors", [])
                    author_str = "; ".join(
                        f"{a.get('last_name', '')}, {a.get('first_name', '')}".strip(", ")
                        for a in authors_list
                    ) if authors_list else None
                    journal_obj = p.get("journal", {})
                    journal_name = journal_obj.get("title") if isinstance(journal_obj, dict) else None
                    dim = {
                        "doi": p.get("doi"),
                        "title": p.get("title"),
                        "authors": author_str,
                        "journal": journal_name,
                        "year": p.get("year"),
                        "volume": p.get("volume"),
                        "issue": p.get("issue"),
                        "pages": p.get("pages"),
                    }
                    meta = enrich(meta, dim)
            elif r:
                logger.warning(f"Dimensions.ai returned HTTP {r.status_code}")
        except Exception as e:
            logger.warning(f"Dimensions.ai error: {e}")

    # ---------- 🔟 Semantic Scholar ----------
    try:
        s2_headers = headers.copy()
        if SEMANTIC_SCHOLAR_API_KEY:
            s2_headers['x-api-key'] = SEMANTIC_SCHOLAR_API_KEY
        r = _request_with_retry(
            f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}"
            "?fields=title,year,venue,url,authors",
            headers=s2_headers,
            max_retries=2,
        )
        if r and r.status_code == 200:
            s = r.json()
            ss = {
                "authors": "; ".join(a.get("name", "") for a in s.get("authors", [])) or None,
                "title": s.get("title"),
                "journal": s.get("venue"),
                "year": s.get("year"),
                "url": s.get("url") or f"https://doi.org/{doi}",
            }
            meta = enrich(meta, ss)
            if is_complete(meta):
                return meta
        elif r:
            logger.warning(f"Semantic Scholar returned HTTP {r.status_code} for DOI {doi}")
    except Exception as e:
        logger.warning(f"Semantic Scholar error for DOI {doi}: {e}")

    # ---------- 1️⃣1️⃣ CORE ----------
    # Disabled by default: title-based search is slow and frequently times out
    # Pass enable_core=True to re-enable
    if enable_core and CORE_API_KEY:
        try:
            search_title = meta.get("title") or ""
            if search_title:
                q = urllib.parse.quote(f'title:"{search_title}"')
                core_headers = {**headers, "Authorization": f"Bearer {CORE_API_KEY}"}
                r = requests.get(
                    f"https://api.core.ac.uk/v3/search/works?q={q}&limit=5",
                    headers=core_headers,
                    timeout=10,
                )
                if r.status_code == 200:
                    results = r.json().get("results", [])
                    for doc in results[:5]:
                        fetched_title = doc.get("title", "")
                        ratio = SequenceMatcher(None, search_title.lower(), fetched_title.lower()).ratio()
                        if ratio >= 0.9:
                            authors_list = doc.get("authors", [])
                            authors = "; ".join(
                                _format_initial(a.get("name", "") if isinstance(a, dict) else str(a))
                                for a in authors_list
                            ) or None
                            journals = doc.get("journals") or []
                            journal_title = journals[0].get("title") if journals else None
                            co = {
                                "authors": authors,
                                "title": fetched_title,
                                "journal": journal_title or doc.get("publisher"),
                                "year": doc.get("yearPublished"),
                                "url": doc.get("downloadUrl") or (doc.get("sourceFulltextUrls") or [None])[0],
                            }
                            meta = enrich(meta, co)
                            break
                    if is_complete(meta):
                        return meta
                else:
                    logger.warning(f"CORE returned HTTP {r.status_code}")
        except Exception as e:
            logger.warning(f"CORE error: {e}")

    # ---------- 5️⃣ DBLP ----------
    try:
        search_title = meta.get("title") or ""
        if search_title:
            q = urllib.parse.quote(search_title)
            r = requests.get(
                f"https://dblp.org/search/publ/api?q={q}&format=json&h=5",
                headers=headers,
                timeout=5,
            )
            if r.status_code == 200:
                hits = r.json().get("result", {}).get("hits", {}).get("hit", [])
                for hit in hits[:5]:
                    info = hit.get("info", {})
                    fetched_title = (info.get("title") or "").rstrip(".")
                    ratio = SequenceMatcher(None, search_title.lower(), fetched_title.lower()).ratio()
                    if ratio >= 0.9:
                        authors_data = info.get("authors", {}).get("author", [])
                        if isinstance(authors_data, dict):
                            authors_data = [authors_data]
                        authors = "; ".join(
                            _format_initial(a.get("text", "") if isinstance(a, dict) else str(a))
                            for a in authors_data
                        ) or None
                        db = {
                            "authors": authors,
                            "title": fetched_title,
                            "journal": info.get("venue"),
                            "volume": info.get("volume"),
                            "pages": info.get("pages"),
                            "year": int(info["year"]) if info.get("year", "").isdigit() else None,
                            "url": info.get("ee") or info.get("url"),
                        }
                        meta = enrich(meta, db)
                        break
                if is_complete(meta):
                    return meta
            else:
                logger.warning(f"DBLP returned HTTP {r.status_code}")
    except Exception as e:
        logger.warning(f"DBLP error: {e}")

    # ---------- 1️⃣2️⃣ OpenCitations Meta ----------
    try:
        r = _request_with_retry(
            f"https://api.opencitations.net/meta/v1/metadata/doi:{doi}",
            headers=headers,
        )
        if r and r.status_code == 200:
            try:
                oc_data = r.json()
            except (json.JSONDecodeError, ValueError):
                oc_data = None
            if oc_data and len(oc_data) > 0:
                oc = oc_data[0]
                # Parse author: "Surname, Given [orcid:... omid:...]; ..." → clean names
                raw_author = oc.get("author", "")
                if raw_author:
                    author_str = "; ".join(
                        re.sub(r'\s*\[.*?\]', '', a).strip()
                        for a in raw_author.split(";")
                        if a.strip()
                    )
                else:
                    author_str = None
                # Parse year from pub_date (format: YYYY-MM-DD or YYYY)
                pub_date = oc.get("pub_date", "")
                year_val = pub_date.split("-")[0] if pub_date else None
                # Parse DOI from id field: "doi:10.xxx omid:... pmid:..."
                oc_doi = None
                for part in oc.get("id", "").split():
                    if part.startswith("doi:"):
                        oc_doi = part[4:]
                        break
                oc_meta = {
                    "doi": oc_doi or doi,
                    "title": oc.get("title"),
                    "authors": author_str,
                    "journal": oc.get("source_title"),
                    "year": year_val,
                    "volume": oc.get("volume"),
                    "issue": oc.get("issue"),
                    "pages": oc.get("page"),
                }
                meta = enrich(meta, oc_meta)
        elif r:
            logger.warning(f"OpenCitations returned HTTP {r.status_code}")
    except Exception as e:
        logger.warning(f"OpenCitations error: {e}")

    # ---------- 1️⃣3️⃣ Europe PMC (last: only returns abbreviated author initials) ----------
    try:
        r = _request_with_retry(
            f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:{doi}&format=json",
        )
        if r and r.status_code == 200:
            data = r.json().get("resultList", {}).get("result", [])
            if data:
                d = data[0]
                ep = {
                    "authors": d.get("authorString"),
                    "title": d.get("title"),
                    "journal": d.get("journalTitle"),
                    "volume": d.get("journalVolume"),
                    "issue": d.get("issue"),
                    "pages": d.get("pageInfo"),
                    "year": d.get("pubYear"),
                    "url": d.get("fullTextUrlList", {}).get("fullTextUrl", [{}])[0].get("url", f"https://doi.org/{doi}"),
                }
                meta = enrich(meta, ep)
                if is_complete(meta):
                    return meta
        elif r:
            logger.warning(f"Europe PMC returned HTTP {r.status_code} for DOI {doi}")
    except Exception as e:
        logger.warning(f"Europe PMC error for DOI {doi}: {e}")

    # ---------- 1️⃣1️⃣ OpenAlex ----------
    try:
        openalex_url = f"https://api.openalex.org/works/https://doi.org/{doi}"
        if OPENALEX_API_KEY:
            openalex_url += f"?api_key={OPENALEX_API_KEY}"
        r = requests.get(openalex_url, headers=headers, timeout=10)
        if r.status_code == 200:
            data = r.json()
            oa = {
                "authors": "; ".join([a["author"]["display_name"] for a in data.get("authorships", [])]) or None,
                "title": data.get("title"),
                "journal": data.get("host_venue", {}).get("display_name"),
                "volume": data.get("biblio", {}).get("volume"),
                "issue": data.get("biblio", {}).get("issue"),
                "pages": data.get("biblio", {}).get("first_page"),
                "year": data.get("publication_year"),
                "url": data.get("host_venue", {}).get("url") or f"https://doi.org/{doi}",
            }
            meta = enrich(meta, oa)
            if is_complete(meta):
                return meta
        else:
            logger.warning(f"OpenAlex returned HTTP {r.status_code} for DOI {doi}")
    except Exception as e:
        logger.warning(f"OpenAlex error for DOI {doi}: {e}")
    time.sleep(delay)

    # ---------- 1️⃣5️⃣ doi.org BibTeX content negotiation (catches non-Crossref registrars) ----------
    try:
        import bibtexparser
        bib_headers = {**headers, "Accept": "application/x-bibtex; charset=utf-8"}
        r = _request_with_retry(f"https://doi.org/{doi}", headers=bib_headers)
        if r and r.status_code == 200 and r.text.strip().startswith("@"):
            # Normalize non-standard month strings before parsing.
            # bibtexparser rejects variants like "july", "june", "sept", "january", etc.
            _MONTH_MAP = {
                "january": "jan", "february": "feb", "march": "mar",
                "april": "apr", "june": "jun", "july": "jul",
                "august": "aug", "september": "sep", "sept": "sep",
                "october": "oct", "november": "nov", "december": "dec",
            }
            def _normalize_bib_months(text):
                def _replace_month(m):
                    val = m.group(1).strip().lower().rstrip(",")
                    return f"month = {{{_MONTH_MAP.get(val, val)}}}"
                return re.sub(
                    r'\bmonth\s*=\s*\{?([a-zA-Z]+)\}?',
                    _replace_month,
                    text,
                    flags=re.IGNORECASE,
                )
            bib_text = _normalize_bib_months(r.text)
            db = bibtexparser.loads(bib_text)
            if db.entries:
                e = db.entries[0]
                # BibTeX uses "and" between authors; convert to "; " separator
                raw_authors = e.get("author", "")
                if raw_authors:
                    bib_authors = "; ".join(
                        _format_initial(a.strip()) for a in raw_authors.split(" and ") if a.strip()
                    ) or None
                else:
                    bib_authors = None
                # BibTeX uses "number" for issue
                bib_year = e.get("year")
                # Normalize en-dash/em-dash in pages to simple hyphen
                bib_pages = e.get("pages", "")
                if bib_pages:
                    bib_pages = bib_pages.replace("\u2013", "-").replace("\u2014", "-")
                bx = {
                    "authors": bib_authors,
                    "title": e.get("title", "").rstrip(".") or None,
                    "journal": e.get("journal"),
                    "volume": e.get("volume"),
                    "issue": e.get("number"),
                    "pages": bib_pages or None,
                    "year": int(bib_year) if bib_year and bib_year.isdigit() else None,
                    "url": e.get("url") or f"https://doi.org/{doi}",
                }
                meta = enrich(meta, bx)
        elif r:
            logger.warning(f"doi.org BibTeX returned HTTP {r.status_code} for DOI {doi}")
    except ImportError:
        logger.debug("bibtexparser not installed, skipping doi.org BibTeX fallback")
    except Exception as e:
        logger.warning(f"doi.org BibTeX error for DOI {doi}: {e}")

    # ---------- Default fallback ----------
    if not meta["url"]:
        meta["url"] = f"https://doi.org/{doi}"
    return meta
