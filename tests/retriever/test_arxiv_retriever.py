"""Tests for ArxivRetriever."""

import sys
import time
from types import SimpleNamespace

import feedparser
import requests

from zotero_arxiv_daily.retriever.arxiv_retriever import (
    ArxivRetriever,
    _extract_text_from_html_worker,
    _run_with_hard_timeout,
    extract_text_from_html,
)
import zotero_arxiv_daily.retriever.arxiv_retriever as arxiv_retriever


def _sleep_and_return(value: str, delay_seconds: float) -> str:
    time.sleep(delay_seconds)
    return value


def _raise_runtime_error() -> None:
    raise RuntimeError("boom")


class FakeResponse:
    def __init__(self, status_code: int = 200, text: str = "<html></html>"):
        self.status_code = status_code
        self.text = text

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            error = requests.HTTPError(f"{self.status_code} response")
            error.response = SimpleNamespace(status_code=self.status_code)
            raise error


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


def test_extract_text_from_html_returns_none_on_404(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(arxiv_retriever, "logger", SimpleNamespace(warning=warnings.append))
    monkeypatch.setattr(arxiv_retriever.requests, "get", lambda *args, **kwargs: FakeResponse(status_code=404))
    monkeypatch.setitem(
        sys.modules,
        "trafilatura",
        SimpleNamespace(extract=lambda *args, **kwargs: "should not be used"),
    )

    paper = SimpleNamespace(title="No HTML paper", entry_id="http://arxiv.org/abs/2604.07328v1")

    assert extract_text_from_html(paper) is None
    assert warnings == []


def test_extract_text_from_html_worker_extracts_text(monkeypatch):
    monkeypatch.setattr(
        arxiv_retriever.requests,
        "get",
        lambda *args, **kwargs: FakeResponse(text="<html><body>paper text</body></html>"),
    )
    monkeypatch.setitem(
        sys.modules,
        "trafilatura",
        SimpleNamespace(extract=lambda *args, **kwargs: "paper text"),
    )

    assert _extract_text_from_html_worker("https://arxiv.org/html/2402.08954v1") == "paper text"


def test_extract_text_from_html_logs_warning_when_no_text_extracted(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(arxiv_retriever, "logger", SimpleNamespace(warning=warnings.append))
    monkeypatch.setattr(
        arxiv_retriever.requests,
        "get",
        lambda *args, **kwargs: FakeResponse(text="<html><body>empty</body></html>"),
    )
    monkeypatch.setitem(sys.modules, "trafilatura", SimpleNamespace(extract=lambda *args, **kwargs: None))

    paper = SimpleNamespace(title="Empty HTML paper", entry_id="https://arxiv.org/abs/2402.08954v1")

    assert extract_text_from_html(paper) is None
    assert "No text extracted" in warnings[0]


def test_extract_text_from_html_logs_warning_on_server_error(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(arxiv_retriever, "logger", SimpleNamespace(warning=warnings.append))
    monkeypatch.setattr(arxiv_retriever.requests, "get", lambda *args, **kwargs: FakeResponse(status_code=500))
    monkeypatch.setitem(
        sys.modules,
        "trafilatura",
        SimpleNamespace(extract=lambda *args, **kwargs: "should not be used"),
    )

    paper = SimpleNamespace(title="Broken HTML paper", entry_id="https://arxiv.org/abs/2402.08954v1")

    assert extract_text_from_html(paper) is None
    assert "500 response" in warnings[0]
