import requests
import time
import urllib.parse
import re
import logging
import os
import html
import unicodedata
from pathlib import Path
from difflib import SequenceMatcher
from fetch_metadata_from_doi import _authors_have_abbreviations, _new_authors_are_better

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


def _title_similarity(query_title, fetched_title, threshold=0.9):
    """Return a similarity score (0-1) if titles match, or 0.0 if below threshold.

    Uses strict SequenceMatcher ratio with a high threshold (0.9) to avoid
    false positives from near-duplicate titles that differ in a single key word
    (e.g. "unit test" vs "system test"). Secondary checks handle subtitle
    variants ("Title" vs "Title: Subtitle") common in replications.
    """
    if not query_title or not fetched_title:
        return 0.0
    a = _normalize_title(query_title)
    b = _normalize_title(fetched_title)
    score = SequenceMatcher(None, a, b).ratio()
    if score >= threshold:
        return score
    # Secondary check: if one title is the main part (before colon) of the other,
    # this handles "Title" vs "Title: A Subtitle" patterns.
    # Require exact containment of the main title, not fuzzy matching.
    a_main = a.split(":")[0].strip()
    b_main = b.split(":")[0].strip()
    if len(a_main) >= 10 and a_main == b_main:
        logger.info(f"Title match via exact main title (score={score:.2f}): '{query_title[:60]}'")
        return max(score, threshold)
    if len(a_main) >= 10 and len(b_main) >= 10:
        # One full title equals the other's main title (before subtitle)
        if a == b_main or b == a_main:
            logger.info(f"Title match via subtitle containment (score={score:.2f}): query='{query_title[:60]}' vs fetched='{fetched_title[:60]}'")
            return max(score, threshold)
    return 0.0


def _titles_match(query_title, fetched_title, threshold=0.9):
    """Check if a fetched title is similar enough to the query title."""
    return _title_similarity(query_title, fetched_title, threshold) > 0.0


def _validate_metadata(fetched, expected_journal=None, expected_year=None, expected_volume=None):
    """
    Validate fetched metadata against expected values.
    Returns True if metadata matches or no expected values provided.
    Used for disambiguating generic titles.
    """
    if not fetched:
        return False

    # If we have expected values, check them
    checks_passed = 0
    checks_total = 0

    if expected_year:
        checks_total += 1
        fetched_year = str(fetched.get('year', '')).strip()
        if fetched_year and str(expected_year).strip() == fetched_year:
            checks_passed += 1
        elif not fetched_year:
            # No year in fetched data, don't penalize
            pass
        else:
            # Year mismatch - fail validation
            return False

    if expected_journal:
        checks_total += 1
        fetched_journal = str(fetched.get('journal', '')).lower().strip()
        expected_j = str(expected_journal).lower().strip()
        # Fuzzy match on journal name (allows "Am Econ Rev" vs "American Economic Review")
        if fetched_journal and (expected_j in fetched_journal or fetched_journal in expected_j):
            checks_passed += 1
        elif not fetched_journal:
            # No journal in fetched data, don't penalize
            pass
        else:
            # Journal mismatch - fail validation
            logger.debug(f"Journal mismatch: expected='{expected_journal}' vs fetched='{fetched.get('journal')}'")
            return False

    if expected_volume:
        checks_total += 1
        fetched_volume = str(fetched.get('volume', '')).strip()
        if fetched_volume and str(expected_volume).strip() == fetched_volume:
            checks_passed += 1
        elif not fetched_volume:
            # No volume in fetched data, don't penalize
            pass
        # Volume mismatch is not a hard failure (volumes can be inconsistent)

    # If we have expected values and at least one matched, consider it valid
    if checks_total > 0 and checks_passed > 0:
        return True
    # If no expected values provided, validation passes
    if checks_total == 0:
        return True

    return False


def normalize_doi(doi):
    """
    Normalize a DOI by removing any URL prefix.
    Handles cases where DOI might already be a full URL.
    Returns just the DOI part (e.g., '10.1234/xyz')
    """
    if not isinstance(doi, str) or not doi.strip():
        return None
    doi = doi.strip()
    # Strip common URL prefixes
    if doi.startswith("http://doi.org/"):
        doi = doi.replace("http://doi.org/", "")
    elif doi.startswith("https://doi.org/"):
        doi = doi.replace("https://doi.org/", "")
    elif doi.startswith("http://dx.doi.org/"):
        doi = doi.replace("http://dx.doi.org/", "")
    elif doi.startswith("https://dx.doi.org/"):
        doi = doi.replace("https://dx.doi.org/", "")
    return doi if doi else None


def _normalize_title(title):
    """Normalize a title for comparison: decode HTML entities, NFKD unicode, collapse whitespace."""
    if not title:
        return ""
    t = html.unescape(title)
    t = unicodedata.normalize("NFKD", t)
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t




def fetch_metadata_from_title(title, email=None, delay=0.2, authors=None,
                             journal=None, year=None, volume=None):
    if email is None:
        email = CONTACT_EMAIL
    """
    Progressive multi-API metadata enrichment starting from a title.
    Crossref → Entrez/PubMed → DataCite → Zenodo → BASE → Scopus → Dimensions.ai → Semantic Scholar → DBLP → CORE → OpenCitations → Europe PMC → OpenAlex
    Attempts to find the DOI first, then uses DOI-based lookups to fill metadata.

    Args:
        authors: Optional author string (e.g. "Nass, C.; Moon, Y.") used to
                 improve Crossref search accuracy via query.author.
        journal: Optional journal name for validation when title is generic.
        year: Optional publication year for validation when title is generic.
        volume: Optional volume number for validation when title is generic.
    """
    if not isinstance(title, str) or not title.strip():
        return None


    title = re.sub(r"\(\s*\d{4}\s*\)", "", title)       # remove "(YYYY)"
    title = re.sub(r"[\s\-\.,:;]+$", "", title).strip()  # trim extra punctuation/ and leading/trailing spaces

    headers = {
        # Pure Chrome-on-Windows user-agent (spoof)
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/117.0.0.0 Safari/537.36"
        )
    }


    meta = {k: None for k in ["doi", "pmid", "authors", "title", "journal", "volume", "issue", "pages", "year", "url"]}

    def enrich(current, new):
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
        for k in m:
            val = m.get(k)
            if val in [None, "", "NaN"]:
                return False
            if k == "authors" and _authors_have_abbreviations(val):
                return False
        return True

    doi = meta.get("doi")
    if not doi:
        # ---------- 2️⃣ Try Crossref title search ----------
        try:
            q = urllib.parse.quote(title)
            cr_url = f"https://api.crossref.org/works?query.title={q}&rows=5"
            if CROSSREF_EMAIL:
                cr_url += f"&mailto={CROSSREF_EMAIL}"
            # Add author filter if available (extract first surname)
            if authors:
                first_author = re.split(r"[;,]", authors)[0].strip()
                # Extract surname: handle "Last, F." and "First Last" formats
                surname = first_author.split(",")[0].strip() if "," in first_author else first_author.split()[-1] if first_author.split() else ""
                if surname and len(surname) >= 2:
                    cr_url += f"&query.author={urllib.parse.quote(surname)}"
            # Add Crossref Plus API key if available
            crossref_headers = headers.copy()
            if CROSSREF_API_KEY:
                crossref_headers['Crossref-Plus-API-Token'] = f'Bearer {CROSSREF_API_KEY}'
            r = _request_with_retry(cr_url, headers=crossref_headers)
            if r and r.status_code == 200:
                items = r.json()["message"].get("items", [])
                best_sim, best_item = 0.0, None
                for item in items[:5]:
                    # Extract metadata for validation
                    item_year = (
                        item.get("published-print", {}).get("date-parts", [[None]])[0][0]
                        or item.get("published-online", {}).get("date-parts", [[None]])[0][0]
                    )
                    candidate_meta = {
                        'journal': (item.get("container-title") or [None])[0],
                        'year': item_year,
                        'volume': item.get("volume"),
                    }
                    # Skip if validation fails
                    if not _validate_metadata(candidate_meta, journal, year, volume):
                        continue
                    fetched_title = (item.get("title") or [None])[0]
                    sim = _title_similarity(title, fetched_title)
                    if sim > best_sim:
                        best_sim, best_item = sim, item
                if best_item:
                    fetched_title = (best_item.get("title") or [None])[0]
                    doi = normalize_doi(best_item.get("DOI"))
                    cr_authors = []
                    for a in best_item.get("author", []):
                        parts = []
                        if "given" in a: parts.append(a["given"])
                        if "family" in a: parts.append(a["family"])
                        name = " ".join(parts).strip()
                        if name:
                            cr_authors.append(name)
                    year = (
                        best_item.get("published-print", {}).get("date-parts", [[None]])[0][0]
                        or best_item.get("published-online", {}).get("date-parts", [[None]])[0][0]
                    )
                    cr = {
                        "doi": doi,
                        "authors": "; ".join(cr_authors) or None,
                        "title": fetched_title,
                        "journal": (best_item.get("container-title") or [None])[0],
                        "volume": best_item.get("volume"),
                        "issue": best_item.get("issue"),
                        "pages": best_item.get("page"),
                        "year": year,
                        "url": f"https://doi.org/{doi}" if doi else None,
                    }
                    meta = enrich(meta, cr)
                    if is_complete(meta):
                        return meta
            elif r:
                logger.warning(f"Crossref returned HTTP {r.status_code} for title search")
        except Exception as e:
            logger.warning(f"Crossref title search error: {e}")

    if not doi:
        # ---------- 3️⃣ NCBI Entrez/PubMed ----------
        try:
            from Bio import Entrez
            Entrez.email = email
            # Add API key if available (increases rate limit from 3/s to 10/s)
            if ENTREZ_API_KEY:
                Entrez.api_key = ENTREZ_API_KEY
            # Strict title field search first
            handle = Entrez.esearch(db="pubmed", term=f"{title}[Title]", retmax=3)
            search_results = Entrez.read(handle)
            handle.close()
            pmids = search_results.get("IdList", [])
            # Fall back to fuzzy full-text search if strict fails
            if not pmids:
                handle = Entrez.esearch(db="pubmed", term=title, retmax=3)
                search_results = Entrez.read(handle)
                handle.close()
                pmids = search_results.get("IdList", [])
            if pmids:
                handle = Entrez.efetch(db="pubmed", id=pmids[0], rettype="xml")
                records = Entrez.read(handle)
                handle.close()
                article = records["PubmedArticle"][0]["MedlineCitation"]["Article"]
                fetched_title = str(article.get("ArticleTitle", ""))
                if _titles_match(title, fetched_title):
                    # Extract DOI from ArticleIdList
                    id_list = records["PubmedArticle"][0].get("PubmedData", {}).get("ArticleIdList", [])
                    entrez_doi = None
                    for aid in id_list:
                        if aid.attributes.get("IdType") == "doi":
                            entrez_doi = normalize_doi(str(aid))
                            break
                    # Extract metadata
                    journal_info = article.get("Journal", {})
                    pub_date = journal_info.get("JournalIssue", {}).get("PubDate", {})
                    authors_list = []
                    for author in article.get("AuthorList", []):
                        last = author.get("LastName", "")
                        fore = author.get("ForeName", "")
                        if last:
                            authors_list.append(f"{fore} {last}".strip())
                    entrez_pmid = str(pmids[0])
                    ez = {
                        "doi": entrez_doi,
                        "pmid": entrez_pmid,
                        "authors": "; ".join(authors_list) or None,
                        "title": fetched_title,
                        "journal": journal_info.get("Title"),
                        "volume": journal_info.get("JournalIssue", {}).get("Volume"),
                        "issue": journal_info.get("JournalIssue", {}).get("Issue"),
                        "pages": article.get("Pagination", {}).get("StartPage") or article.get("Pagination", {}).get("MedlinePgn"),
                        "year": pub_date.get("Year"),
                        "url": f"https://doi.org/{entrez_doi}" if entrez_doi else f"https://pubmed.ncbi.nlm.nih.gov/{entrez_pmid}/",
                    }
                    meta = enrich(meta, ez)
                    doi = meta.get("doi")
                    if is_complete(meta):
                        return meta
        except ImportError:
            logger.debug("Biopython not installed, skipping Entrez search")
        except Exception as e:
            logger.warning(f"Entrez/PubMed search error: {e}")
        time.sleep(delay)

    # ---------- 5️⃣ DataCite (if DOI found) ----------
    if doi:
        try:
            r = _request_with_retry(f"https://api.datacite.org/dois/{doi.lower()}", headers=headers)
            if r and r.status_code == 200:
                d = r.json().get("data", {}).get("attributes", {})
                authors = []
                for a in d.get("creators", []):
                    name = a.get("name") or f"{a.get('givenName','')} {a.get('familyName','')}".strip()
                    if name:
                        authors.append(name)
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

    # ---------- 5️⃣b Zenodo ----------
    # Covers preprints, conference papers, software, datasets — corpus not in other APIs
    try:
        search_title = meta.get("title") or title
        q = urllib.parse.quote(f'title:"{search_title}"')
        r = _request_with_retry(
            f"https://zenodo.org/api/records?q={q}&size=5&sort=bestmatch",
            headers=headers,
        )
        if r and r.status_code == 200:
            hits = r.json().get("hits", {}).get("hits", [])
            best_sim, best_hit = 0.0, None
            for hit in hits[:5]:
                fetched_title = hit.get("metadata", {}).get("title", "")
                candidate_meta = {
                    'year': hit.get("metadata", {}).get("publication_date", "")[:4] or None,
                }
                if not _validate_metadata(candidate_meta, journal, year, volume):
                    continue
                sim = _title_similarity(search_title, fetched_title)
                if sim > best_sim:
                    best_sim, best_hit = sim, hit
            if best_hit:
                md = best_hit.get("metadata", {})
                creators = md.get("creators", [])
                zen_authors = "; ".join(c.get("name", "") for c in creators if c.get("name")) or None
                zen_doi = normalize_doi(md.get("doi") or best_hit.get("doi"))
                if not doi and zen_doi:
                    doi = zen_doi
                pub_date = md.get("publication_date", "")
                zen_year = int(pub_date[:4]) if pub_date and len(pub_date) >= 4 and pub_date[:4].isdigit() else None
                journal_obj = md.get("journal", {})
                zen_journal = journal_obj.get("title") if isinstance(journal_obj, dict) else None
                zn = {
                    "doi": zen_doi,
                    "authors": zen_authors,
                    "title": md.get("title"),
                    "journal": zen_journal,
                    "year": zen_year,
                    "url": f"https://doi.org/{zen_doi}" if zen_doi else best_hit.get("links", {}).get("html"),
                }
                meta = enrich(meta, zn)
                if is_complete(meta):
                    return meta
        elif r:
            logger.warning(f"Zenodo returned HTTP {r.status_code} for title search")
    except Exception as e:
        logger.warning(f"Zenodo title search error: {e}")

    # ---------- 7️⃣ BASE (Bielefeld Academic Search Engine) ----------
    try:
        search_title = meta.get("title") or title
        q = urllib.parse.quote(search_title)
        r = _request_with_retry(
            f"https://api.base-search.net/cgi-bin/BaseHttpSearchInterface.fcgi"
            f"?func=PerformSearch&query=dctitle:{q}&format=json&hits=5",
            headers=headers,
        )
        if r and r.status_code == 200:
            results = r.json().get("response", {}).get("docs", [])
            best_sim, best_doc = 0.0, None
            for doc in results[:5]:
                fetched_title = doc.get("dctitle", "")
                if isinstance(fetched_title, list):
                    fetched_title = fetched_title[0] if fetched_title else ""
                candidate_meta = {
                    'journal': doc.get("dcsource"),
                    'year': str(doc.get("dcyear", "")),
                }
                if not _validate_metadata(candidate_meta, journal, year, volume):
                    continue
                sim = _title_similarity(search_title, fetched_title)
                if sim > best_sim:
                    best_sim, best_doc = sim, doc
            if best_doc:
                fetched_title = best_doc.get("dctitle", "")
                if isinstance(fetched_title, list):
                    fetched_title = fetched_title[0] if fetched_title else ""
                fetched_doi = normalize_doi(best_doc.get("dcdoi"))
                if not doi and fetched_doi:
                    doi = fetched_doi
                authors_raw = best_doc.get("dcCreator") or best_doc.get("dccreator") or []
                if isinstance(authors_raw, str):
                    authors_raw = [authors_raw]
                authors = "; ".join(a for a in authors_raw if a) or None
                base_year = None
                dcdate = best_doc.get("dcyear") or best_doc.get("dcdate", "")
                if isinstance(dcdate, list):
                    dcdate = dcdate[0] if dcdate else ""
                dcdate = str(dcdate)
                if dcdate and len(dcdate) >= 4 and dcdate[:4].isdigit():
                    base_year = int(dcdate[:4])
                ba = {
                    "doi": fetched_doi,
                    "authors": authors,
                    "title": fetched_title,
                    "journal": best_doc.get("dcsource") or best_doc.get("dcpublisher"),
                    "year": base_year,
                    "url": best_doc.get("dclink") or best_doc.get("dcidentifier") or (f"https://doi.org/{fetched_doi}" if fetched_doi else None),
                }
                meta = enrich(meta, ba)
                if is_complete(meta):
                    return meta
        elif r:
            logger.warning(f"BASE returned HTTP {r.status_code} for title search")
    except Exception as e:
        logger.warning(f"BASE title search error: {e}")

    # ---------- 8️⃣ Scopus / Elsevier ----------
    if SCOPUS_API_KEY:
        try:
            scopus_headers = {**headers, "X-ELS-APIKey": SCOPUS_API_KEY, "Accept": "application/json"}
            q = urllib.parse.quote(title)
            scopus_url = f"https://api.elsevier.com/content/search/scopus?query=TITLE({q})&count=5"
            if year:
                scopus_url += f"+AND+PUBYEAR+IS+{year}"
            r = _request_with_retry(scopus_url, headers=scopus_headers)
            if r and r.status_code == 200:
                results = r.json().get("search-results", {}).get("entry", [])
                for entry in results[:5]:
                    fetched_title = entry.get("dc:title", "")
                    sim = _title_similarity(title, fetched_title)
                    if sim > 0:  # _title_similarity already enforces 0.9 threshold
                        # Get DOI
                        entry_doi = entry.get("prism:doi")
                        # Get full metadata via abstract retrieval if DOI available
                        if entry_doi:
                            r2 = _request_with_retry(
                                f"https://api.elsevier.com/content/abstract/doi/{entry_doi}",
                                headers=scopus_headers,
                            )
                            if r2 and r2.status_code == 200:
                                resp = r2.json().get("abstracts-retrieval-response", {})
                                coredata = resp.get("coredata", {})
                                authors_obj = resp.get("authors", {}).get("author", [])
                                if isinstance(authors_obj, dict):
                                    authors_obj = [authors_obj]
                                author_str = "; ".join(
                                    f"{a.get('ce:given-name', '')} {a.get('ce:surname', '')}".strip()
                                    for a in authors_obj
                                    if a.get('ce:surname')
                                ) or None
                                cover_date = coredata.get("prism:coverDate", "")
                                year_val = int(cover_date[:4]) if cover_date and len(cover_date) >= 4 and cover_date[:4].isdigit() else None
                                sc = {
                                    "doi": entry_doi,
                                    "title": coredata.get("dc:title") or fetched_title,
                                    "authors": author_str,
                                    "journal": coredata.get("prism:publicationName"),
                                    "volume": coredata.get("prism:volume"),
                                    "issue": coredata.get("prism:issueIdentifier"),
                                    "pages": coredata.get("prism:pageRange"),
                                    "year": year_val,
                                    "url": f"https://doi.org/{entry_doi}",
                                }
                                if _validate_metadata(sc, journal, year, volume):
                                    meta = enrich(meta, sc)
                                    if not doi and entry_doi:
                                        doi = entry_doi
                                    if is_complete(meta):
                                        return meta
                        else:
                            # Fallback: use search result metadata
                            cover_date = entry.get("prism:coverDate", "")
                            year_val = int(cover_date[:4]) if cover_date and len(cover_date) >= 4 and cover_date[:4].isdigit() else None
                            sc = {
                                "title": fetched_title,
                                "authors": entry.get("dc:creator"),
                                "journal": entry.get("prism:publicationName"),
                                "volume": entry.get("prism:volume"),
                                "issue": entry.get("prism:issueIdentifier"),
                                "pages": entry.get("prism:pageRange"),
                                "year": year_val,
                            }
                            if _validate_metadata(sc, journal, year, volume):
                                meta = enrich(meta, sc)
                                if is_complete(meta):
                                    return meta
                        break
            elif r:
                logger.warning(f"Scopus returned HTTP {r.status_code} for title search")
        except Exception as e:
            logger.warning(f"Scopus title search error: {e}")

    # ---------- 🔟 Dimensions.ai ----------
    if DIMENSIONS_API_KEY:
        try:
            import json as json_lib
            dim_headers = {**headers, "Authorization": f"Bearer {DIMENSIONS_API_KEY}"}
            # Search by title (and optionally year/authors for better matching)
            search_parts = [f'title="{title}"']
            if year:
                search_parts.append(f'year={year}')
            search_clause = " and ".join(search_parts)
            query = f'search publications where {search_clause} return publications[doi+title+authors+journal+year+volume+issue+pages] limit 5'
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
                    # Find best title match
                    best_sim = 0.0
                    best_p = None
                    for p in pubs:
                        p_title = p.get("title", "")
                        if p_title:
                            sim = _title_similarity(title, p_title)
                            if sim > best_sim:
                                best_sim = sim
                                best_p = p
                    if best_p and best_sim > 0:  # _title_similarity already enforces 0.85 threshold
                        authors_list = best_p.get("authors", [])
                        author_str = "; ".join(
                            f"{a.get('last_name', '')}, {a.get('first_name', '')}".strip(", ")
                            for a in authors_list
                        ) if authors_list else None
                        journal_obj = best_p.get("journal", {})
                        journal_name = journal_obj.get("title") if isinstance(journal_obj, dict) else None
                        dim = {
                            "doi": best_p.get("doi"),
                            "title": best_p.get("title"),
                            "authors": author_str,
                            "journal": journal_name,
                            "year": best_p.get("year"),
                            "volume": best_p.get("volume"),
                            "issue": best_p.get("issue"),
                            "pages": best_p.get("pages"),
                        }
                        if _validate_metadata(dim, journal, year, volume):
                            meta = enrich(meta, dim)
            elif r:
                logger.warning(f"Dimensions.ai returned HTTP {r.status_code} for title search")
        except Exception as e:
            logger.warning(f"Dimensions.ai title search error: {e}")

    # ---------- 🔟 Semantic Scholar (if DOI or title available) ----------
    try:
        s2_headers = headers.copy()
        if SEMANTIC_SCHOLAR_API_KEY:
            s2_headers['x-api-key'] = SEMANTIC_SCHOLAR_API_KEY
        if doi:
            url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}?fields=title,year,venue,url,authors"
            r = _request_with_retry(url, headers=s2_headers, max_retries=2)
            if r and r.status_code == 200:
                s = r.json()
                fetched_title = s.get("title")
                ss = {
                    "doi": doi,
                    "authors": "; ".join(a.get("name", "") for a in s.get("authors", [])) or None,
                    "title": fetched_title,
                    "journal": s.get("venue"),
                    "year": s.get("year"),
                    "url": s.get("url") or f"https://doi.org/{doi}",
                }
                meta = enrich(meta, ss)
            elif r:
                logger.warning(f"Semantic Scholar returned HTTP {r.status_code} for DOI query")
        else:
            q = urllib.parse.quote(title)
            url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={q}&limit=5&fields=title,year,venue,url,authors,externalIds"
            r = _request_with_retry(url, headers=s2_headers, max_retries=2)
            if r and r.status_code == 200:
                data = r.json()
                best_sim, best_s = 0.0, None
                for s in data.get("data", [])[:5]:
                    sim = _title_similarity(title, s.get("title"))
                    if sim > best_sim:
                        best_sim, best_s = sim, s
                if best_s:
                    ext_ids = best_s.get("externalIds", {}) or {}
                    fetched_doi = normalize_doi(ext_ids.get("DOI"))
                    s2_pmid = ext_ids.get("PubMed")
                    doi = fetched_doi
                    s2_url = best_s.get("url")
                    if not s2_url:
                        if doi:
                            s2_url = f"https://doi.org/{doi}"
                        elif s2_pmid:
                            s2_url = f"https://pubmed.ncbi.nlm.nih.gov/{s2_pmid}/"
                    ss = {
                        "doi": doi,
                        "pmid": s2_pmid,
                        "authors": "; ".join(a.get("name", "") for a in best_s.get("authors", [])) or None,
                        "title": best_s.get("title"),
                        "journal": best_s.get("venue"),
                        "year": best_s.get("year"),
                        "url": s2_url,
                    }
                    meta = enrich(meta, ss)
            elif r:
                logger.warning(f"Semantic Scholar returned HTTP {r.status_code} for title query")
    except Exception as e:
        logger.warning(f"Semantic Scholar error: {e}")

    # ---------- 6️⃣ DBLP ----------
    try:
        search_title = meta.get("title") or title
        q = urllib.parse.quote(search_title)
        r = requests.get(
            f"https://dblp.org/search/publ/api?q={q}&format=json&h=5",
            headers=headers,
            timeout=5,
        )
        if r.status_code == 200:
            hits = r.json().get("result", {}).get("hits", {}).get("hit", [])
            best_sim, best_info = 0.0, None
            for hit in hits[:5]:
                info = hit.get("info", {})
                fetched_title = (info.get("title") or "").rstrip(".")
                candidate_meta = {
                    'journal': info.get("venue"),
                    'year': info.get("year"),
                    'volume': info.get("volume"),
                }
                if not _validate_metadata(candidate_meta, journal, year, volume):
                    continue
                sim = _title_similarity(search_title, fetched_title)
                if sim > best_sim:
                    best_sim, best_info = sim, info
            if best_info:
                fetched_title = (best_info.get("title") or "").rstrip(".")
                fetched_doi = normalize_doi(best_info.get("doi"))
                if not doi and fetched_doi:
                    doi = fetched_doi
                authors_data = best_info.get("authors", {}).get("author", [])
                if isinstance(authors_data, dict):
                    authors_data = [authors_data]
                authors = "; ".join(
                    a.get("text", "") if isinstance(a, dict) else str(a)
                    for a in authors_data
                ) or None
                db = {
                    "doi": fetched_doi,
                    "authors": authors,
                    "title": fetched_title,
                    "journal": best_info.get("venue"),
                    "volume": best_info.get("volume"),
                    "pages": best_info.get("pages"),
                    "year": int(best_info["year"]) if best_info.get("year", "").isdigit() else None,
                    "url": best_info.get("ee") or best_info.get("url") or (f"https://doi.org/{fetched_doi}" if fetched_doi else None),
                }
                meta = enrich(meta, db)
                if is_complete(meta):
                    return meta
        else:
            logger.warning(f"DBLP returned HTTP {r.status_code} for title search")
    except Exception as e:
        logger.warning(f"DBLP title search error: {e}")

    # ---------- 1️⃣1️⃣ CORE ----------
    if CORE_API_KEY:
        try:
            search_title = meta.get("title") or title
            q = urllib.parse.quote(f'title:"{search_title}"')
            core_headers = {**headers, "Authorization": f"Bearer {CORE_API_KEY}"}
            r = requests.get(
                f"https://api.core.ac.uk/v3/search/works?q={q}&limit=5",
                headers=core_headers,
                timeout=10,
            )
            if r.status_code == 200:
                results = r.json().get("results", [])
                best_sim, best_doc = 0.0, None
                for doc in results[:5]:
                    fetched_title = doc.get("title", "")
                    candidate_meta = {
                        'journal': doc.get("publisher"),
                        'year': doc.get("yearPublished"),
                    }
                    if not _validate_metadata(candidate_meta, journal, year, volume):
                        continue
                    sim = _title_similarity(search_title, fetched_title)
                    if sim > best_sim:
                        best_sim, best_doc = sim, doc
                if best_doc:
                    fetched_doi = normalize_doi(best_doc.get("doi"))
                    if not doi and fetched_doi:
                        doi = fetched_doi
                    authors_list = best_doc.get("authors", [])
                    authors = "; ".join(
                        a.get("name", "") if isinstance(a, dict) else str(a)
                        for a in authors_list
                    ) or None
                    journals = best_doc.get("journals") or []
                    journal_title = journals[0].get("title") if journals else None
                    co = {
                        "doi": fetched_doi,
                        "authors": authors,
                        "title": best_doc.get("title"),
                        "journal": journal_title or best_doc.get("publisher"),
                        "year": best_doc.get("yearPublished"),
                        "url": best_doc.get("downloadUrl") or (best_doc.get("sourceFulltextUrls") or [None])[0] or (f"https://doi.org/{fetched_doi}" if fetched_doi else None),
                    }
                    meta = enrich(meta, co)
                    if is_complete(meta):
                        return meta
            else:
                logger.warning(f"CORE returned HTTP {r.status_code} for title search")
        except Exception as e:
            logger.warning(f"CORE title search error: {e}")

    # ---------- 1️⃣2️⃣ OpenCitations Meta ----------
    # OpenCitations only works if we already have a DOI
    if doi:
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
                    if _validate_metadata(oc_meta, journal, year, volume):
                        meta = enrich(meta, oc_meta)
            elif r:
                logger.warning(f"OpenCitations returned HTTP {r.status_code}")
        except Exception as e:
            logger.warning(f"OpenCitations error: {e}")

    # ---------- 1️⃣3️⃣ Europe PMC (last: only returns abbreviated author initials) ----------
    try:
        q = urllib.parse.quote(meta.get("title") or title)
        r = _request_with_retry(
            f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query={q}&format=json&pageSize=5",
        )
        if r and r.status_code == 200:
            results = r.json().get("resultList", {}).get("result", [])
            best_sim, best_d = 0.0, None
            for d in results[:5]:
                candidate_meta = {
                    'journal': d.get("journalTitle"),
                    'year': d.get("pubYear"),
                    'volume': d.get("journalVolume"),
                }
                if not _validate_metadata(candidate_meta, journal, year, volume):
                    continue
                sim = _title_similarity(title, d.get("title"))
                if sim > best_sim:
                    best_sim, best_d = sim, d
            if best_d:
                fetched_title = best_d.get("title")
                normalized_doi = normalize_doi(best_d.get("doi"))
                epmc_pmid = best_d.get("pmid")
                epmc_url = best_d.get("fullTextUrlList", {}).get("fullTextUrl", [{}])[0].get("url")
                if not epmc_url and epmc_pmid:
                    epmc_url = f"https://pubmed.ncbi.nlm.nih.gov/{epmc_pmid}/"
                ep = {
                    "doi": normalized_doi,
                    "pmid": epmc_pmid,
                    "authors": best_d.get("authorString"),
                    "title": fetched_title,
                    "journal": best_d.get("journalTitle"),
                    "volume": best_d.get("journalVolume"),
                    "issue": best_d.get("issue"),
                    "pages": best_d.get("pageInfo"),
                    "year": best_d.get("pubYear"),
                    "url": epmc_url,
                }
                meta = enrich(meta, ep)
                if is_complete(meta):
                    return meta
        elif r:
            logger.warning(f"Europe PMC returned HTTP {r.status_code} for title search")
    except Exception as e:
        logger.warning(f"Europe PMC title search error: {e}")

    # ---------- 1️⃣0️⃣ OpenAlex search by title ----------
    try:
        q = urllib.parse.quote(title)
        openalex_url = f"https://api.openalex.org/works?filter=title.search:{q}&per_page=5"
        if OPENALEX_API_KEY:
            openalex_url += f"&api_key={OPENALEX_API_KEY}"
        r = requests.get(openalex_url, headers=headers, timeout=10)
        if r.status_code == 200:
            results = r.json().get("results", [])
            best_sim, best_data = 0.0, None
            for data in results[:5]:
                # Extract metadata for validation
                candidate_meta = {
                    'journal': data.get("host_venue", {}).get("display_name"),
                    'year': data.get("publication_year"),
                    'volume': data.get("biblio", {}).get("volume"),
                }
                # Skip if validation fails (journal/year/volume mismatch)
                if not _validate_metadata(candidate_meta, journal, year, volume):
                    continue
                sim = _title_similarity(title, data.get("title"))
                if sim > best_sim:
                    best_sim, best_data = sim, data
            if best_data:
                fetched_title = best_data.get("title")
                oa_doi = normalize_doi(best_data.get("doi"))
                if oa_doi and not doi:
                    doi = oa_doi
                oa = {
                    "doi": oa_doi,
                    "authors": "; ".join([a["author"]["display_name"] for a in best_data.get("authorships", [])]) or None,
                    "title": fetched_title,
                    "journal": best_data.get("host_venue", {}).get("display_name"),
                    "volume": best_data.get("biblio", {}).get("volume"),
                    "issue": best_data.get("biblio", {}).get("issue"),
                    "pages": best_data.get("biblio", {}).get("first_page"),
                    "year": best_data.get("publication_year"),
                    "url": f"https://doi.org/{oa_doi}" if oa_doi else best_data.get("host_venue", {}).get("url"),
                }
                meta = enrich(meta, oa)
                if is_complete(meta):
                    return meta
        else:
            logger.warning(f"OpenAlex returned HTTP {r.status_code} for title search")
    except Exception as e:
        logger.warning(f"OpenAlex title search error: {e}")
    time.sleep(delay)

    # ---------- Default fallback ----------
    if not meta.get("url"):
        if meta.get("doi"):
            meta["url"] = f"https://doi.org/{meta['doi']}"
        elif meta.get("pmid"):
            meta["url"] = f"https://pubmed.ncbi.nlm.nih.gov/{meta['pmid']}/"
    return meta
