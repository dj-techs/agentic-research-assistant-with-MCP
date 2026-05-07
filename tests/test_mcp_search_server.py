"""Tests for the search MCP server.

These run against the fixture mode so they don't need network access.
"""

from __future__ import annotations

import json

import pytest

from research_assistant.mcp_servers.search_server import arxiv_search, web_search


@pytest.fixture(autouse=True)
def fixture_mode(monkeypatch):
    monkeypatch.setenv("SEARCH_MODE", "fixture")
    yield


def test_web_search_returns_fixture_hits():
    hits = json.loads(web_search("Transformer architecture", num_results=2))
    assert isinstance(hits, list) and len(hits) >= 1
    blob = " ".join(h.get("title", "") + h.get("snippet", "") for h in hits).lower()
    assert "transformer" in blob or "vaswani" in blob


def test_arxiv_search_returns_bert():
    hits = json.loads(arxiv_search("BERT", max_results=1))
    assert isinstance(hits, list) and len(hits) == 1
    assert "BERT" in hits[0]["title"]


def test_unknown_query_returns_empty_in_fixture_mode():
    hits = json.loads(web_search("zzz nonsense query that nobody asks", num_results=3))
    assert hits == []
