"""Tests for ArxivRetriever."""

import time
from types import SimpleNamespace
from datetime import datetime, timezone

import feedparser

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


def test_arxiv_retriever_falls_back_when_rss_is_empty(config, monkeypatch):
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)

    empty_feed = feedparser.FeedParserDict(
        feed=feedparser.FeedParserDict(title="empty"),
        entries=[],
    )
    monkeypatch.setattr(feedparser, "parse", lambda *_args, **_kwargs: empty_feed)

    fake_results = [
        SimpleNamespace(
            title="Primary category materials paper",
            authors=[SimpleNamespace(name="Test Author")],
            summary="Test abstract",
            pdf_url="https://arxiv.org/pdf/2607.00001",
            entry_id="https://arxiv.org/abs/2607.00001",
            published=datetime(2026, 7, 5, 12, 0, tzinfo=timezone.utc),
            primary_category="cond-mat.mtrl-sci",
            source_url=lambda: "https://arxiv.org/e-print/2607.00001",
        ),
        SimpleNamespace(
            title="Cross listed paper",
            authors=[SimpleNamespace(name="Test Author")],
            summary="Test abstract",
            pdf_url="https://arxiv.org/pdf/2607.00002",
            entry_id="https://arxiv.org/abs/2607.00002",
            published=datetime(2026, 7, 5, 9, 0, tzinfo=timezone.utc),
            primary_category="cond-mat.stat-mech",
            source_url=lambda: "https://arxiv.org/e-print/2607.00002",
        ),
        SimpleNamespace(
            title="Older paper should be dropped",
            authors=[SimpleNamespace(name="Test Author")],
            summary="Test abstract",
            pdf_url="https://arxiv.org/pdf/2607.00003",
            entry_id="https://arxiv.org/abs/2607.00003",
            published=datetime(2026, 7, 3, 20, 0, tzinfo=timezone.utc),
            primary_category="cond-mat.mtrl-sci",
            source_url=lambda: "https://arxiv.org/e-print/2607.00003",
        ),
    ]

    class FakeClient:
        def __init__(self, **kw):
            pass

        def results(self, search):
            assert search.query == "cat:cond-mat.mtrl-sci OR cat:physics.app-ph"
            return iter(fake_results)

    monkeypatch.setattr(arxiv_retriever.arxiv, "Client", FakeClient)
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_html", lambda paper: None)
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_pdf", lambda paper: None)
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_tar", lambda paper: None)

    config.source.arxiv.category = ["cond-mat.mtrl-sci", "physics.app-ph"]
    config.source.arxiv.include_cross_list = False

    retriever = ArxivRetriever(config)
    papers = retriever.retrieve_papers()

    assert [paper.title for paper in papers] == ["Primary category materials paper"]


def test_arxiv_retriever_fallback_includes_cross_lists_when_enabled(config, monkeypatch):
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)

    empty_feed = feedparser.FeedParserDict(
        feed=feedparser.FeedParserDict(title="empty"),
        entries=[],
    )
    monkeypatch.setattr(feedparser, "parse", lambda *_args, **_kwargs: empty_feed)

    fake_results = [
        SimpleNamespace(
            title="Primary category materials paper",
            authors=[SimpleNamespace(name="Test Author")],
            summary="Test abstract",
            pdf_url="https://arxiv.org/pdf/2607.00001",
            entry_id="https://arxiv.org/abs/2607.00001",
            published=datetime(2026, 7, 5, 12, 0, tzinfo=timezone.utc),
            primary_category="cond-mat.mtrl-sci",
            source_url=lambda: "https://arxiv.org/e-print/2607.00001",
        ),
        SimpleNamespace(
            title="Cross listed paper",
            authors=[SimpleNamespace(name="Test Author")],
            summary="Test abstract",
            pdf_url="https://arxiv.org/pdf/2607.00002",
            entry_id="https://arxiv.org/abs/2607.00002",
            published=datetime(2026, 7, 5, 9, 0, tzinfo=timezone.utc),
            primary_category="cond-mat.stat-mech",
            source_url=lambda: "https://arxiv.org/e-print/2607.00002",
        ),
    ]

    class FakeClient:
        def __init__(self, **kw):
            pass

        def results(self, search):
            return iter(fake_results)

    monkeypatch.setattr(arxiv_retriever.arxiv, "Client", FakeClient)
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_html", lambda paper: None)
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_pdf", lambda paper: None)
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_tar", lambda paper: None)

    config.source.arxiv.category = ["cond-mat.mtrl-sci", "physics.app-ph"]
    config.source.arxiv.include_cross_list = True

    retriever = ArxivRetriever(config)
    papers = retriever.retrieve_papers()

    assert [paper.title for paper in papers] == [
        "Primary category materials paper",
        "Cross listed paper",
    ]


def test_extract_text_from_html_returns_none_for_missing_arxiv_html(monkeypatch):
    requested_urls = []

    class FakeResponse:
        status_code = 404
        text = ""

        def raise_for_status(self):
            raise AssertionError("404 should be handled without raising")

    def fake_get(url, timeout):
        requested_urls.append(url)
        return FakeResponse()

    paper = SimpleNamespace(
        title="Paper without arXiv HTML",
        entry_id="http://arxiv.org/abs/2607.12213v1",
    )

    monkeypatch.setattr(arxiv_retriever.requests, "get", fake_get)

    assert arxiv_retriever.extract_text_from_html(paper) is None
    assert requested_urls == ["https://arxiv.org/html/2607.12213v1"]


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
