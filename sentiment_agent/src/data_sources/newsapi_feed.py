"""Fetch financial news headlines via NewsAPI.org.

Supports generic global queries and India-specific source targeting
(Economic Times, MoneyControl, etc.) via the `domains` parameter.
"""

import os
import requests
from datetime import datetime, timedelta
from src.config.settings import NEWSAPI_KEY


BASE_URL = "https://newsapi.org/v2/everything"

# ---------------------------------------------------------------------------
# Query templates
# ---------------------------------------------------------------------------

DEFAULT_QUERIES = [
    "RBI AND (repo OR rate OR policy OR MPC)",
    "SBI OR \"State Bank of India\" OR PNB OR \"Punjab National Bank\"",
    "Bajaj Finance OR \"Shriram Finance\" OR \"Cholamandalam Finance\"",
    "NBFC AND (India OR RBI OR regulation)",
    "PSU bank AND (India OR merger OR NPA OR profit)",
]

# Tighter queries when we know the source is an Indian financial outlet
INDIA_FINANCE_QUERIES = [
    "(RBI OR MPC OR repo rate OR monetary policy)",
    "(SBI OR PNB OR Bank of Baroda OR Canara Bank OR Union Bank)",
    "(Bajaj Finance OR Shriram Finance OR Cholamandalam OR M&M Finance)",
    "(NBFC OR shadow bank OR microfinance) AND (loan OR credit OR NPA)",
    "(PSU bank OR public sector bank) AND (profit OR loss OR NPA OR merger)",
    "(budget OR finance minister) AND (bank OR NBFC OR lending)",
]

# ---------------------------------------------------------------------------
# India-specific financial news domains supported by NewsAPI
# ---------------------------------------------------------------------------

INDIA_FINANCE_DOMAINS = [
    "economictimes.indiatimes.com",   # Economic Times
    "moneycontrol.com",               # MoneyControl
    "business-standard.com",            # Business Standard
    "livemint.com",                   # Mint
    "financialexpress.com",           # Financial Express
    "cnbctv18.com",                   # CNBC-TV18
    "ndtv.com",                       # NDTV (business section mixed)
]


def _build_params(
    query: str,
    from_date: str,
    to_date: str,
    language: str,
    page_size: int,
    domains: list[str] | None = None,
) -> dict:
    """Build request params dict for NewsAPI /v2/everything."""
    params = {
        "q": query,
        "from": from_date,
        "to": to_date,
        "language": language,
        "sortBy": "relevancy",
        "pageSize": min(page_size, 100),
        "apiKey": NEWSAPI_KEY,
    }
    if domains:
        # NewsAPI accepts comma-separated domains
        params["domains"] = ",".join(domains)
    return params


def fetch_headlines(
    query: str | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
    language: str = "en",
    page_size: int = 20,
    domains: list[str] | None = None,
) -> list[dict]:
    """
    Fetch headlines from NewsAPI.

    Args:
        query: search query string. If None, uses DEFAULT_QUERIES.
        from_date: YYYY-MM-DD. Defaults to 2 days ago.
        to_date: YYYY-MM-DD. Defaults to today.
        language: article language code.
        page_size: max 100 per request.
        domains: restrict to specific source domains (e.g. ["moneycontrol.com"]).

    Returns:
        List of article dicts: [{"title", "description", "url", "publishedAt", "source"}, ...]
    """
    if not NEWSAPI_KEY:
        raise RuntimeError("NEWSAPI_KEY not set. Check .env file.")

    if not query:
        query = " OR ".join(DEFAULT_QUERIES)

    if not from_date:
        from_date = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
    if not to_date:
        to_date = datetime.now().strftime("%Y-%m-%d")

    params = _build_params(query, from_date, to_date, language, page_size, domains)

    resp = requests.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if data.get("status") != "ok":
        raise RuntimeError(f"NewsAPI error: {data.get('message', 'unknown')}")

    articles = data.get("articles", [])
    results = []
    for a in articles:
        url = a.get("url", "")
        domain = url.split("/")[2] if "//" in url else ""
        results.append({
            "title": a.get("title", ""),
            "description": a.get("description", ""),
            "url": url,
            "publishedAt": a.get("publishedAt", ""),
            "source": a.get("source", {}).get("name", ""),
            "domain": domain,
        })

    return results


def fetch_india_finance_headlines(
    query: str | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
    page_size: int = 20,
    sources: list[str] | None = None,
) -> list[dict]:
    """
    Fetch headlines from India-specific financial news sources.

    Args:
        query: optional custom query. If None, uses INDIA_FINANCE_QUERIES.
        sources: subset of domain names to target. If None, uses all
                 INDIA_FINANCE_DOMAINS.

    Returns:
        Same shape as fetch_headlines.
    """
    if not query:
        query = " OR ".join(INDIA_FINANCE_QUERIES)

    domains = sources if sources is not None else INDIA_FINANCE_DOMAINS
    return fetch_headlines(
        query=query,
        from_date=from_date,
        to_date=to_date,
        language="en",
        page_size=page_size,
        domains=domains,
    )


def fetch_and_analyze(
    query: str | None = None,
    narrative: bool = False,
    india_only: bool = False,
    sources: list[str] | None = None,
) -> list[dict]:
    """Fetch headlines and run each through the LangGraph pipeline."""
    from src.graph.builder import get_graph

    if india_only:
        headlines = fetch_india_finance_headlines(query=query, sources=sources)
    else:
        headlines = fetch_headlines(query=query)

    graph = get_graph()

    results = []
    for h in headlines:
        title = h["title"]
        if not title:
            continue
        out = graph.invoke(
            {
                "query": f"Analyze: {title}",
                "headline": title,
                "narrative_flag": narrative,
                "moderation": {},
                "routing": {},
                "macro_context": None,
                "psu_results": [],
                "nbfc_results": [],
                "report": None,
                "narrative": None,
                "status": "PENDING",
                "reasons": [],
            }
        )
        results.append(
            {
                "headline": title,
                "url": h["url"],
                "source": h["source"],
                "publishedAt": h["publishedAt"],
                "pipeline": out,
            }
        )

    return results
