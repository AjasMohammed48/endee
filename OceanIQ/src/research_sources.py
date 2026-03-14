import re
import requests

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; OceanIQ/1.0)"}
TIMEOUT = 8


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text).strip()


def fetch_wikipedia(query: str, limit: int = 3) -> list:
    results = []
    try:
        r = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action":   "query",
                "list":     "search",
                "srsearch": query,
                "srlimit":  limit,
                "format":   "json",
                "origin":   "*",
            },
            headers=HEADERS,
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        pages = r.json().get("query", {}).get("search", [])

        for page in pages:
            title = page.get("title", "")
            slug  = requests.utils.quote(title.replace(" ", "_"))

            try:
                sr = requests.get(
                    f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(title)}",
                    headers=HEADERS,
                    timeout=TIMEOUT,
                )
                sr.raise_for_status()
                data    = sr.json()
                summary = _strip_html(data.get("extract", ""))
                url     = data.get("content_urls", {}).get("desktop", {}).get("page", "")
                if not url:
                    url = f"https://en.wikipedia.org/wiki/{slug}"
            except Exception:
                summary = _strip_html(page.get("snippet", ""))
                url     = f"https://en.wikipedia.org/wiki/{slug}"

            if title and summary:
                results.append({
                    "title":   title,
                    "summary": summary[:400],
                    "url":     url,
                    "source":  "Wikipedia",
                })

    except Exception:
        pass

    return results


def fetch_semantic_scholar(query: str, limit: int = 3) -> list:
    results = []
    try:
        r = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={
                "query":  query,
                "limit":  limit,
                "fields": "title,abstract,authors,year,externalIds,url",
            },
            headers=HEADERS,
            timeout=TIMEOUT,
        )
        r.raise_for_status()

        for p in r.json().get("data", []):
            authors = ", ".join(a.get("name", "") for a in (p.get("authors") or [])[:3])
            doi     = (p.get("externalIds") or {}).get("DOI", "")
            link    = p.get("url") or (f"https://doi.org/{doi}" if doi else "")
            results.append({
                "title":    p.get("title", ""),
                "abstract": _strip_html((p.get("abstract") or "")[:400]),
                "authors":  authors,
                "year":     p.get("year", ""),
                "url":      link,
                "source":   "Semantic Scholar",
            })

    except Exception:
        pass

    return results


def fetch_pubmed(query: str, limit: int = 3) -> list:
    results = []
    try:
        r = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={
                "db":      "pubmed",
                "term":    query,
                "retmax":  limit,
                "retmode": "json",
            },
            headers=HEADERS,
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        ids = r.json().get("esearchresult", {}).get("idlist", [])

        if not ids:
            return []

        sr = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
            params={
                "db":      "pubmed",
                "id":      ",".join(ids),
                "retmode": "json",
            },
            headers=HEADERS,
            timeout=TIMEOUT,
        )
        sr.raise_for_status()
        docs = sr.json().get("result", {})

        for pid in ids:
            doc   = docs.get(pid, {})
            title = doc.get("title", "")
            if title:
                results.append({
                    "title":    _strip_html(title),
                    "abstract": _strip_html(doc.get("source", "")),
                    "authors":  ", ".join(a.get("name", "") for a in (doc.get("authors") or [])[:3]),
                    "year":     (doc.get("pubdate") or "")[:4],
                    "url":      f"https://pubmed.ncbi.nlm.nih.gov/{pid}/",
                    "source":   "PubMed",
                })

    except Exception:
        pass

    return results


def fetch_patents(query: str) -> dict:
    return {
        "url":   f"https://patents.google.com/?q={requests.utils.quote(query)}",
        "query": query,
    }


def fetch_all_sources(query: str, source_flags: dict) -> dict:
    results = {}

    if source_flags.get("wikipedia", True):
        results["wikipedia"] = fetch_wikipedia(query)

    if source_flags.get("semantic_scholar", True):
        results["semantic_scholar"] = fetch_semantic_scholar(query)

    if source_flags.get("pubmed", True):
        results["pubmed"] = fetch_pubmed(query)

    if source_flags.get("patents", True):
        results["patents"] = fetch_patents(query)

    return results
