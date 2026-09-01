"""Tests for ChemrxivRetriever (Crossref-backed)."""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
import requests
from omegaconf import open_dict

from zotero_arxiv_daily.retriever import get_retriever_cls
from zotero_arxiv_daily.retriever.chemrxiv_retriever import ChemrxivRetriever
from tests.canned_responses import (
    SAMPLE_CHEMRXIV_API_RESPONSE,
    _chemrxiv_item,
    _chemrxiv_response,
)


FIXED_NOW = datetime(2026, 3, 2, 12, 0, 0, tzinfo=timezone.utc)


class _FixedDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return FIXED_NOW if tz is None else FIXED_NOW.astimezone(tz)


@pytest.fixture()
def fixed_now(monkeypatch):
    monkeypatch.setattr("zotero_arxiv_daily.retriever.chemrxiv_retriever.datetime", _FixedDatetime)


def _patch_pages(monkeypatch, pages):
    """Patch requests.get to serve ``pages`` (list of Crossref responses) keyed by ``offset``."""
    calls = []

    def _patched(url, **kwargs):
        assert "api.crossref.org" in url
        params = kwargs.get("params", {})
        calls.append(params)
        offset = int(params.get("offset", 0))
        rows = int(params.get("rows", 100))
        idx = offset // rows
        page = pages[idx] if idx < len(pages) else _chemrxiv_response([])
        resp = SimpleNamespace(status_code=200, raise_for_status=lambda: None)
        resp.json = lambda: page
        return resp

    monkeypatch.setattr(requests, "get", _patched)
    return calls


def _configure(config, include_new_versions=False):
    with open_dict(config.source):
        config.source.chemrxiv = {"include_new_versions": include_new_versions}
    return ChemrxivRetriever(config)


def test_chemrxiv_is_registered():
    assert get_retriever_cls("chemrxiv") is ChemrxivRetriever


def test_chemrxiv_retrieve_filters_by_date_and_version(config, monkeypatch, fixed_now):
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)
    calls = _patch_pages(monkeypatch, [SAMPLE_CHEMRXIV_API_RESPONSE])
    retriever = _configure(config)
    papers = retriever.retrieve_papers()
    # v2 revision dropped; old paper outside 24h window dropped.
    assert [p.title for p in papers] == ["A chemrxiv paper", "Another chemrxiv paper"]
    assert calls[0]["sort"] == "created"
    assert calls[0]["order"] == "desc"
    assert calls[0]["rows"] == 100
    assert calls[0]["filter"] == "type:posted-content,from-created-date:2026-03-01"


def test_chemrxiv_include_new_versions(config, monkeypatch, fixed_now):
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)
    _patch_pages(monkeypatch, [SAMPLE_CHEMRXIV_API_RESPONSE])
    retriever = _configure(config, include_new_versions=True)
    papers = retriever.retrieve_papers()
    assert [p.title for p in papers] == [
        "A chemrxiv paper",
        "A revised chemrxiv paper",
        "Another chemrxiv paper",
    ]


def test_chemrxiv_paginates_until_cutoff(config, monkeypatch, fixed_now):
    monkeypatch.setattr(ChemrxivRetriever, "page_size", 2)
    recent = [
        _chemrxiv_item(f"10.26434/chemrxiv.{i}/v1", f"Paper {i}", f"2026-03-02T0{9 - i}:00:00Z", [("A", "B")])
        for i in range(4)
    ]
    old = _chemrxiv_item("10.26434/chemrxiv.old/v1", "Old", "2026-02-01T00:00:00Z", [("A", "B")])
    pages = [
        _chemrxiv_response(recent[:2], total=5),
        _chemrxiv_response(recent[2:], total=5),
        _chemrxiv_response([old], total=5),
        _chemrxiv_response([old], total=5),  # must never be requested
    ]
    calls = _patch_pages(monkeypatch, pages)
    retriever = _configure(config)
    raw = retriever._retrieve_raw_papers()
    assert [r["title"][0] for r in raw] == ["Paper 0", "Paper 1", "Paper 2", "Paper 3"]
    assert [c["offset"] for c in calls] == [0, 2, 4]


def test_chemrxiv_stops_on_short_page(config, monkeypatch, fixed_now):
    calls = _patch_pages(monkeypatch, [SAMPLE_CHEMRXIV_API_RESPONSE])
    retriever = _configure(config)
    retriever._retrieve_raw_papers()
    assert len(calls) == 1


def test_chemrxiv_empty_response(config, monkeypatch, fixed_now):
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)
    _patch_pages(monkeypatch, [_chemrxiv_response([])])
    retriever = _configure(config)
    assert retriever.retrieve_papers() == []


def test_chemrxiv_convert_to_paper(config):
    retriever = _configure(config)
    raw = SAMPLE_CHEMRXIV_API_RESPONSE["message"]["items"][0]
    paper = retriever.convert_to_paper(raw)
    assert paper.source == "chemrxiv"
    assert paper.title == "A chemrxiv paper"
    assert paper.authors == ["Jane Smith", "Alan Doe"]
    # JATS tags stripped, entities unescaped, whitespace collapsed
    assert paper.abstract == "We study RuO 2 & friends."
    assert paper.url == "https://chemrxiv.org/doi/full/10.26434/chemrxiv.15007618/v1"
    assert paper.pdf_url == "https://chemrxiv.org/doi/pdf/10.26434/chemrxiv.15007618/v1"
    assert paper.full_text is None


def test_chemrxiv_convert_to_paper_fallbacks(config):
    retriever = _configure(config)
    raw = dict(SAMPLE_CHEMRXIV_API_RESPONSE["message"]["items"][0])
    raw.pop("resource")
    raw.pop("abstract")
    raw["author"] = [{"name": "Some Consortium"}, {"given": "Only", "family": "Person"}]
    paper = retriever.convert_to_paper(raw)
    assert paper.url == "https://doi.org/10.26434/chemrxiv.15007618/v1"
    assert paper.abstract == ""
    assert paper.authors == ["Some Consortium", "Only Person"]


def test_chemrxiv_convert_to_paper_without_title_is_skipped(config):
    retriever = _configure(config)
    raw = dict(SAMPLE_CHEMRXIV_API_RESPONSE["message"]["items"][0])
    raw["title"] = []
    assert retriever.convert_to_paper(raw) is None


def test_chemrxiv_version_detection(config):
    retriever = _configure(config)
    assert not retriever._is_new_version("10.26434/chemrxiv.15007618/v1")
    assert retriever._is_new_version("10.26434/chemrxiv.15007618/v2")
    assert retriever._is_new_version("10.26434/chemrxiv-2023-abcde-v3")
    assert not retriever._is_new_version("10.26434/chemrxiv-2023-abcde")


def test_chemrxiv_debug_truncates(config, monkeypatch, fixed_now):
    items = [
        _chemrxiv_item(f"10.26434/chemrxiv.{i}/v1", f"Paper {i}", "2026-03-02T09:00:00Z", [("A", "B")])
        for i in range(15)
    ]
    _patch_pages(monkeypatch, [_chemrxiv_response(items)])
    config.executor.debug = True
    retriever = _configure(config)
    assert len(retriever._retrieve_raw_papers()) == 10
