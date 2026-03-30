import time
from types import SimpleNamespace
import feedparser
import io
from urllib.error import HTTPError

from omegaconf import open_dict

from zotero_arxiv_daily.retriever.arxiv_retriever import ArxivRetriever, _run_with_hard_timeout
from zotero_arxiv_daily.retriever.base import BaseRetriever, register_retriever
from zotero_arxiv_daily.protocol import Paper
import zotero_arxiv_daily.retriever.arxiv_retriever as arxiv_retriever


def _sleep_and_return(value: str, delay_seconds: float) -> str:
    time.sleep(delay_seconds)
    return value



def _raise_runtime_error() -> None:
    raise RuntimeError("boom")



def test_arxiv_retriever(config, monkeypatch):

    parsed_result = feedparser.parse("tests/retriever/arxiv_rss_example.xml")
    raw_parser = feedparser.parse
    def mock_feedparser_parse(url):
        if url == f"https://rss.arxiv.org/atom/{'+'.join(config.source.arxiv.category)}":
            return parsed_result
        return raw_parser(url)
    monkeypatch.setattr(feedparser, "parse", mock_feedparser_parse)

    retriever = ArxivRetriever(config)
    papers = retriever.retrieve_papers()
    parsed_results = [i for i in parsed_result.entries if i.get("arxiv_announce_type", "new") == 'new']
    assert len(papers) == len(parsed_results)
    paper_titles = [i.title for i in papers]
    parsed_titles = [i.title for i in parsed_results]
    assert set(paper_titles) == set(parsed_titles)


@register_retriever("failing_test")
class FailingTestRetriever(BaseRetriever):
    def _retrieve_raw_papers(self) -> list[dict[str, str]]:
        return [
            {"title": "good paper", "mode": "ok"},
            {"title": "bad paper", "mode": "fail"},
        ]

    def convert_to_paper(self, raw_paper: dict[str, str]) -> Paper | None:
        if raw_paper["mode"] == "fail":
            raise HTTPError(
                url="https://example.com/paper.pdf",
                code=404,
                msg="not found",
                hdrs=None,
                fp=io.BufferedReader(io.BytesIO(b"missing")),
            )
        return Paper(
            source=self.name,
            title=raw_paper["title"],
            authors=[],
            abstract="",
            url=f"https://example.com/{raw_paper['mode']}",
        )


@register_retriever("serial_test")
class SerialTestRetriever(BaseRetriever):
    def __init__(self, config, seen_titles: list[str]):
        super().__init__(config)
        self.seen_titles = seen_titles

    def _retrieve_raw_papers(self) -> list[dict[str, str]]:
        return [
            {"title": "paper 1"},
            {"title": "paper 2"},
            {"title": "paper 3"},
        ]

    def convert_to_paper(self, raw_paper: dict[str, str]) -> Paper:
        self.seen_titles.append(raw_paper["title"])
        return Paper(
            source=self.name,
            title=raw_paper["title"],
            authors=[],
            abstract="",
            url=f"https://example.com/{raw_paper['title']}",
        )



def test_retrieve_papers_skips_conversion_errors(config):
    with open_dict(config.source):
        config.source.failing_test = {}

    retriever = FailingTestRetriever(config)

    papers = retriever.retrieve_papers()

    assert [paper.title for paper in papers] == ["good paper"]


def test_retrieve_papers_runs_serially(config):
    with open_dict(config.source):
        config.source.serial_test = {}

    seen_titles: list[str] = []
    retriever = SerialTestRetriever(config, seen_titles)

    papers = retriever.retrieve_papers()

    expected_titles = ["paper 1", "paper 2", "paper 3"]
    assert seen_titles == expected_titles
    assert [paper.title for paper in papers] == expected_titles



def test_run_with_hard_timeout_returns_value():
    result = _run_with_hard_timeout(
        _sleep_and_return,
        ("done", 0.01),
        timeout=1,
        operation="test operation",
        paper_title="paper",
    )

    assert result == "done"



def test_run_with_hard_timeout_returns_none_on_timeout(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(arxiv_retriever, "logger", SimpleNamespace(warning=warnings.append))

    result = _run_with_hard_timeout(
        _sleep_and_return,
        ("done", 1.0),
        timeout=0.01,
        operation="test operation",
        paper_title="paper",
    )

    assert result is None
    assert warnings == ["test operation timed out for paper after 0.01 seconds"]



def test_run_with_hard_timeout_returns_none_on_failure(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(arxiv_retriever, "logger", SimpleNamespace(warning=warnings.append))

    result = _run_with_hard_timeout(
        _raise_runtime_error,
        (),
        timeout=1,
        operation="test operation",
        paper_title="paper",
    )

    assert result is None
    assert warnings == ["test operation failed for paper: RuntimeError: boom"]
