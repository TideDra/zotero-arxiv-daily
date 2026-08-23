"""Tests for ArxivRetriever."""

from datetime import datetime
import time
from types import SimpleNamespace

import feedparser

from zotero_arxiv_daily.protocol import Paper
from zotero_arxiv_daily.retriever.arxiv_retriever import ArxivRetriever, _run_with_hard_timeout
import zotero_arxiv_daily.retriever.arxiv_retriever as arxiv_retriever


def _sleep_and_return(value: str, delay_seconds: float) -> str:
    time.sleep(delay_seconds)
    return value


def _raise_runtime_error() -> None:
    raise RuntimeError("boom")


def test_arxiv_retriever(config, mock_feedparser, monkeypatch):
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)

    # The RSS fixture gives us paper IDs.  After feedparser, the code calls
    # arxiv.Client().results(search) which makes real HTTP requests.  We mock
    # the arxiv Client so the test stays offline.
    new_entries = [
        e for e in mock_feedparser.entries
        if e.get("arxiv_announce_type", "new") == "new"
    ]
    paper_ids = [e.id.removeprefix("oai:arXiv.org:") for e in new_entries]

    # Build fake ArxivResult-like objects matching each RSS entry
    fake_results = []
    for entry in new_entries:
        pid = entry.id.removeprefix("oai:arXiv.org:")
        fake_results.append(SimpleNamespace(
            title=entry.title,
            authors=[SimpleNamespace(name="Test Author")],
            summary="Test abstract",
            pdf_url=f"https://arxiv.org/pdf/{pid}",
            entry_id=f"https://arxiv.org/abs/{pid}",
            source_url=lambda pid=pid: f"https://arxiv.org/e-print/{pid}",
        ))

    class FakeClient:
        def __init__(self, **kw):
            pass
        def results(self, search):
            return iter(fake_results)

    monkeypatch.setattr(arxiv_retriever.arxiv, "Client", FakeClient)

    # Skip file downloads in convert_to_paper
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_html", lambda paper: None)
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_pdf", lambda paper: None)
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_tar", lambda paper: None)

    retriever = ArxivRetriever(config)
    papers = retriever.retrieve_papers()

    assert len(papers) == len(new_entries)
    assert set(p.title for p in papers) == set(e.title for e in new_entries)


def test_run_with_hard_timeout_returns_value():
    result = _run_with_hard_timeout(
        _sleep_and_return, ("done", 0.01), timeout=1, operation="test op", paper_title="paper"
    )
    assert result == "done"


def test_run_with_hard_timeout_returns_none_on_timeout(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(arxiv_retriever, "logger", SimpleNamespace(warning=warnings.append))
    result = _run_with_hard_timeout(
        _sleep_and_return, ("done", 1.0), timeout=0.01, operation="test op", paper_title="paper"
    )
    assert result is None
    assert "timed out" in warnings[0]


def test_run_with_hard_timeout_returns_none_on_failure(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(arxiv_retriever, "logger", SimpleNamespace(warning=warnings.append))
    result = _run_with_hard_timeout(
        _raise_runtime_error, (), timeout=1, operation="test op", paper_title="paper"
    )
    assert result is None
    assert "boom" in warnings[0]


def test_enrich_ads_paper_replaces_ads_text_with_arxiv_content(monkeypatch):
    raw = SimpleNamespace(
        title="Open version",
        authors=[SimpleNamespace(name="A. Author")],
        summary="Open arXiv abstract",
        pdf_url="https://arxiv.org/pdf/2608.12345",
        entry_id="https://arxiv.org/abs/2608.12345v2",
        published=datetime(2026, 8, 20),
        categories=["astro-ph.GA"],
        doi=None,
        get_short_id=lambda: "2608.12345v2",
    )
    paper = Paper(
        source="ads",
        title="ADS record",
        authors=["A. Author"],
        abstract="ADS-only text must be replaced",
        url="https://ui.adsabs.harvard.edu/abs/example/abstract",
        external_ids={"ads": "example", "arxiv": "2608.12345"},
        content_source="ads",
        remote_processing_allowed=False,
    )
    monkeypatch.setattr(arxiv_retriever, "fetch_arxiv_results_by_ids", lambda ids: [raw])
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_tar", lambda result: "Open arXiv full text")
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_html", lambda result: None)
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_pdf", lambda result: None)

    arxiv_retriever.enrich_papers_from_arxiv([paper])

    assert paper.abstract == "Open arXiv abstract"
    assert paper.title == "Open version"
    assert paper.full_text == "Open arXiv full text"
    assert paper.content_source == "arxiv"
    assert paper.remote_processing_allowed is True
