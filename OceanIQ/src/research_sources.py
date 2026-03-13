import requests
import threading
from typing import Dict, Any


# ─────────────────────────────────────────────
# WIKIPEDIA
# ─────────────────────────────────────────────
def fetch_wikipedia(query: str, limit: int = 3) -> list:
    try:
        # Search for page titles
        search_url = "https://en.wikipedia.org/w/api.php"
        search_params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": limit,
            "format": "json",
        }
        r = requests.get(search_url, params=search_params, timeout=8)
        r.raise_for_status()
        titles = [item["title"] for item in r.json().get("query", {}).get("search", [])]

        results = []
        for title in titles:
            # Get summary for each title
            summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(title)}"
            sr = requests.get(summary_url, timeout=8)
            if sr.status_code == 200:
                data = sr.json()
                results.append({
                    "title": data.get("title", title),
                    "summary": data.get("extract", "")[:400],
                    "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                })
        return results
    except Exception as e:
        print(f"[research_sources] Wikipedia error: {e}")
        return []


# ─────────────────────────────────────────────
# SEMANTIC SCHOLAR
# ─────────────────────────────────────────────
def fetch_semantic_scholar(query: str, limit: int = 3) -> list:
    try:
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "limit": limit,
            "fields": "title,abstract,authors,year,externalIds,openAccessPdf",
        }
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        papers = r.json().get("data", [])

        results = []
        for p in papers:
            authors = ", ".join(a.get("name", "") for a in p.get("authors", [])[:3])
            doi = p.get("externalIds", {}).get("DOI", "")
            pdf = p.get("openAccessPdf", {})
            pdf_url = pdf.get("url", "") if pdf else ""
            doi_url = f"https://doi.org/{doi}" if doi else ""
            link = pdf_url or doi_url or ""

            results.append({
                "title": p.get("title", ""),
                "abstract": (p.get("abstract") or "")[:400],
                "authors": authors,
                "year": p.get("year", ""),
                "url": link,
            })
        return results
    except Exception as e:
        print(f"[research_sources] Semantic Scholar error: {e}")
        return []


# ─────────────────────────────────────────────
# PUBMED
# ─────────────────────────────────────────────
def fetch_pubmed(query: str, limit: int = 3) -> list:
    try:
        # Step 1: search for IDs
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        search_params = {
            "db": "pubmed",
            "term": query,
            "retmax": limit,
            "retmode": "json",
        }
        r = requests.get(search_url, params=search_params, timeout=8)
        r.raise_for_status()
        ids = r.json().get("esearchresult", {}).get("idlist", [])

        if not ids:
            return []

        # Step 2: fetch summaries
        summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        summary_params = {
            "db": "pubmed",
            "id": ",".join(ids),
            "retmode": "json",
        }
        sr = requests.get(summary_url, params=summary_params, timeout=8)
        sr.raise_for_status()
        result_data = sr.json().get("result", {})

        results = []
        for pmid in ids:
            item = result_data.get(pmid, {})
            title = item.get("title", "")
            authors = ", ".join(
                a.get("name", "") for a in item.get("authors", [])[:3]
            )
            pub_date = item.get("pubdate", "")[:4]
            results.append({
                "title": title,
                "abstract": "",  # Summary API doesn't return abstract; link opens it
                "authors": authors,
                "year": pub_date,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            })
        return results
    except Exception as e:
        print(f"[research_sources] PubMed error: {e}")
        return []


# ─────────────────────────────────────────────
# GOOGLE PATENTS (link only — no paid API)
# ─────────────────────────────────────────────
def fetch_patents(query: str) -> dict:
    encoded = requests.utils.quote(query)
    return {
        "url": f"https://patents.google.com/?q={encoded}",
        "label": f"Search Google Patents for \"{query}\"",
    }


# ─────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────
def fetch_all_sources(query: str, flags: Dict[str, bool] = None) -> Dict[str, Any]:
    """
    Fetch results from all enabled web sources in parallel threads.

    flags dict keys:
        wikipedia, semantic_scholar, pubmed, patents
    All default to True if flags is None.
    """
    if flags is None:
        flags = {
            "wikipedia": True,
            "semantic_scholar": True,
            "pubmed": True,
            "patents": True,
        }

    results: Dict[str, Any] = {}
    lock = threading.Lock()

    def run(key, fn, *args):
        try:
            data = fn(*args)
            with lock:
                results[key] = data
        except Exception as e:
            print(f"[research_sources] Thread error for {key}: {e}")
            with lock:
                results[key] = [] if key != "patents" else {}

    threads = []

    if flags.get("wikipedia", True):
        t = threading.Thread(target=run, args=("wikipedia", fetch_wikipedia, query))
        threads.append(t)

    if flags.get("semantic_scholar", True):
        t = threading.Thread(target=run, args=("semantic_scholar", fetch_semantic_scholar, query))
        threads.append(t)

    if flags.get("pubmed", True):
        t = threading.Thread(target=run, args=("pubmed", fetch_pubmed, query))
        threads.append(t)

    if flags.get("patents", True):
        t = threading.Thread(target=run, args=("patents", fetch_patents, query))
        threads.append(t)

    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=12)

    return results