"""Retriever-specific fixtures."""

import feedparser
import pytest


@pytest.fixture()
def mock_feedparser(monkeypatch):
    """Patch feedparser.parse to return the local RSS fixture for arXiv URLs.

    The arxiv library passes bytes (response.content) to feedparser.parse,
    so we check for both str and bytes URL types.

    Returns the parsed result so tests can assert against it.
    """
    parsed = feedparser.parse("tests/retriever/arxiv_rss_example.xml")
    raw_parse = feedparser.parse

    def _patched(url_or_bytes, *args, **kwargs):
        target = url_or_bytes
        if isinstance(target, bytes):
            target = target.decode("utf-8", errors="ignore")
        if isinstance(target, str) and "rss.arxiv.org" in target:
            return parsed
        return raw_parse(url_or_bytes, *args, **kwargs)

    monkeypatch.setattr(feedparser, "parse", _patched)
    return parsed


@pytest.fixture()
def mock_biorxiv_api(monkeypatch):
    """Patch requests.get to return the canned bioRxiv API response."""
    import requests
    from types import SimpleNamespace

    from tests.canned_responses import SAMPLE_BIORXIV_API_RESPONSE

    original_get = requests.get

    def _patched(url, **kwargs):
        if "api.biorxiv.org" in url:
            resp = SimpleNamespace()
            resp.status_code = 200
            resp.json = lambda: SAMPLE_BIORXIV_API_RESPONSE
            resp.raise_for_status = lambda: None
            return resp
        return original_get(url, **kwargs)

    monkeypatch.setattr(requests, "get", _patched)
