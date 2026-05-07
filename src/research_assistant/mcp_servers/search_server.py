"""Search MCP server.

Exposes three tools over stdio:

* ``web_search``  - DuckDuckGo HTML scrape (no API key required).
* ``arxiv_search`` - Real arXiv Atom feed.
* ``fetch_url``   - Fetch a URL and return cleaned text.

Set ``SEARCH_MODE=fixture`` to load deterministic fixtures from
``evals/fixtures/search.json`` instead of hitting the network. This makes
evaluation reproducible.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from urllib.parse import quote_plus, urlparse

import feedparser
import httpx
from bs4 import BeautifulSoup
from mcp.server.fastmcp import FastMCP

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "evals" / "fixtures" / "search.json"

mcp = FastMCP("search")


def _load_fixtures() -> dict:
    if not FIXTURES.exists():
        return {}
    return json.loads(FIXTURES.read_text())


def _fixture_lookup(kind: str, query: str) -> list[dict] | None:
    data = _load_fixtures().get(kind, {})
    # Match by case-insensitive substring on a key
    q = query.lower().strip()
    for key, value in data.items():
        if key.lower() in q or q in key.lower():
            return value
    return None


@mcp.tool()
def web_search(query: str, num_results: int = 5) -> str:
    """Search the web. Returns a JSON list of {title, url, snippet}."""
    if os.getenv("SEARCH_MODE") == "fixture":
        hits = _fixture_lookup("web", query) or []
        return json.dumps(hits[:num_results])

    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    headers = {"User-Agent": "Mozilla/5.0 (research-assistant)"}
    try:
        resp = httpx.get(url, headers=headers, timeout=10.0, follow_redirects=True)
        resp.raise_for_status()
    except httpx.HTTPError as e:
        return json.dumps({"error": f"web_search failed: {e}"})

    soup = BeautifulSoup(resp.text, "html.parser")
    results: list[dict] = []
    for r in soup.select("div.result")[:num_results]:
        a = r.select_one("a.result__a")
        snippet_el = r.select_one(".result__snippet")
        if not a:
            continue
        results.append(
            {
                "title": a.get_text(strip=True),
                "url": a.get("href", ""),
                "snippet": snippet_el.get_text(strip=True) if snippet_el else "",
            }
        )
    return json.dumps(results)


@mcp.tool()
def arxiv_search(query: str, max_results: int = 5) -> str:
    """Search arXiv. Returns JSON list of {title, authors, summary, url, published}."""
    if os.getenv("SEARCH_MODE") == "fixture":
        hits = _fixture_lookup("arxiv", query) or []
        return json.dumps(hits[:max_results])

    url = (
        "http://export.arxiv.org/api/query"
        f"?search_query=all:{quote_plus(query)}&max_results={max_results}"
    )
    try:
        resp = httpx.get(url, timeout=15.0)
        resp.raise_for_status()
    except httpx.HTTPError as e:
        return json.dumps({"error": f"arxiv_search failed: {e}"})

    feed = feedparser.parse(resp.text)
    results = [
        {
            "title": e.title,
            "authors": [a.name for a in getattr(e, "authors", [])],
            "summary": e.summary.strip().replace("\n", " "),
            "url": e.id,
            "published": getattr(e, "published", ""),
        }
        for e in feed.entries
    ]
    return json.dumps(results)


@mcp.tool()
def fetch_url(url: str, max_chars: int = 4000) -> str:
    """Fetch a URL and return readable text (truncated to ``max_chars``)."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return json.dumps({"error": "Only http(s) URLs are allowed."})

    if os.getenv("SEARCH_MODE") == "fixture":
        hits = _fixture_lookup("fetch", url) or []
        if hits:
            text = hits[0].get("text", "")
            return json.dumps({"url": url, "text": text[:max_chars]})
        return json.dumps({"url": url, "text": ""})

    try:
        resp = httpx.get(
            url,
            timeout=15.0,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (research-assistant)"},
        )
        resp.raise_for_status()
    except httpx.HTTPError as e:
        return json.dumps({"error": f"fetch_url failed: {e}"})

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = " ".join(soup.get_text(separator=" ").split())
    return json.dumps({"url": url, "text": text[:max_chars]})


if __name__ == "__main__":
    # FastMCP.run() is blocking and uses stdio by default.
    try:
        mcp.run()
    except KeyboardInterrupt:
        sys.exit(0)
