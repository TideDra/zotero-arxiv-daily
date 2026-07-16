"""Tests for BaseRetriever: error handling, serial execution, registration."""

import io
from types import SimpleNamespace
from urllib.error import HTTPError

from omegaconf import open_dict

from zotero_arxiv_daily.retriever.base import BaseRetriever, register_retriever, get_retriever_cls
from zotero_arxiv_daily.protocol import Paper


# ---------------------------------------------------------------------------
# Test retrievers — migrated from test_arxiv_retriever.py
# ---------------------------------------------------------------------------


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
        return [{"title": "paper 1"}, {"title": "paper 2"}, {"title": "paper 3"}]

    def convert_to_paper(self, raw_paper: dict[str, str]) -> Paper:
        self.seen_titles.append(raw_paper["title"])
        return Paper(
            source=self.name,
            title=raw_paper["title"],
            authors=[],
            abstract="",
            url=f"https://example.com/{raw_paper['title']}",
        )


@register_retriever("none_test")
class NoneRetriever(BaseRetriever):
    def _retrieve_raw_papers(self):
        return [{"title": "a"}, {"title": "b"}]

    def convert_to_paper(self, raw_paper):
        return None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_retrieve_papers_skips_conversion_errors(config, monkeypatch):
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)
    with open_dict(config.source):
        config.source.failing_test = {}
    retriever = FailingTestRetriever(config)
    papers = retriever.retrieve_papers()
    assert [p.title for p in papers] == ["good paper"]


def test_retrieve_papers_runs_serially(config, monkeypatch):
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)
    with open_dict(config.source):
        config.source.serial_test = {}
    seen: list[str] = []
    retriever = SerialTestRetriever(config, seen)
    papers = retriever.retrieve_papers()
    assert seen == ["paper 1", "paper 2", "paper 3"]
    assert [p.title for p in papers] == ["paper 1", "paper 2", "paper 3"]


def test_retrieve_papers_skips_none_results(config, monkeypatch):
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)
    with open_dict(config.source):
        config.source.none_test = {}
    retriever = NoneRetriever(config)
    papers = retriever.retrieve_papers()
    assert papers == []


def test_retrieve_papers_empty_raw(config, monkeypatch):
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)

    @register_retriever("empty_test")
    class EmptyRetriever(BaseRetriever):
        def _retrieve_raw_papers(self):
            return []
        def convert_to_paper(self, raw_paper):
            return None

    with open_dict(config.source):
        config.source.empty_test = {}
    retriever = EmptyRetriever(config)
    papers = retriever.retrieve_papers()
    assert papers == []


def test_get_retriever_cls_unknown():
    import pytest
    with pytest.raises(ValueError, match="not found"):
        get_retriever_cls("nonexistent_retriever_xyz")
