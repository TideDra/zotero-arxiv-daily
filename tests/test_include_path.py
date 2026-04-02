from datetime import datetime
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from zotero_arxiv_daily.executor import Executor, normalize_path_patterns
from zotero_arxiv_daily.protocol import CorpusPaper


def test_normalize_path_patterns_rejects_single_string_for_include_path():
    with pytest.raises(TypeError, match="config.zotero.include_path must be a list of glob patterns or null"):
        normalize_path_patterns("2026/survey/**", "include_path")


def test_normalize_path_patterns_accepts_list_config_for_include_path():
    include_path = OmegaConf.create(["2026/survey/**", "2026/reading-group/**"])

    assert normalize_path_patterns(include_path, "include_path") == [
        "2026/survey/**",
        "2026/reading-group/**",
    ]


def test_filter_corpus_matches_any_path_against_any_pattern():
    executor = Executor.__new__(Executor)
    executor.config = SimpleNamespace(
        zotero=SimpleNamespace(include_path=["2026/survey/**", "2026/reading-group/**"])
    )
    executor.include_path_patterns = normalize_path_patterns(executor.config.zotero.include_path, "include_path")
    executor.ignore_path_patterns = None

    corpus = [
        CorpusPaper(
            title="Survey Paper",
            abstract="",
            added_date=datetime(2026, 1, 1),
            paths=["2026/survey/topic-a", "archive/misc"],
        ),
        CorpusPaper(
            title="Reading Group Paper",
            abstract="",
            added_date=datetime(2026, 1, 2),
            paths=["notes/inbox", "2026/reading-group/week-1"],
        ),
        CorpusPaper(
            title="Excluded Paper",
            abstract="",
            added_date=datetime(2026, 1, 3),
            paths=["2025/other/topic"],
        ),
    ]

    filtered = executor.filter_corpus(corpus)

    assert [paper.title for paper in filtered] == ["Survey Paper", "Reading Group Paper"]


def test_normalize_path_patterns_rejects_single_string_for_ignore_path():
    with pytest.raises(TypeError, match="config.zotero.ignore_path must be a list of glob patterns or null"):
        normalize_path_patterns("archive/**", "ignore_path")


def test_normalize_path_patterns_accepts_list_config_for_ignore_path():
    ignore_path = OmegaConf.create(["archive/**", "2025/**"])

    assert normalize_path_patterns(ignore_path, "ignore_path") == ["archive/**", "2025/**"]


def test_normalize_path_patterns_accepts_empty_list():
    assert normalize_path_patterns([], "ignore_path") == []


def test_filter_corpus_excludes_papers_matching_ignore_path():
    executor = Executor.__new__(Executor)
    executor.include_path_patterns = None
    executor.ignore_path_patterns = normalize_path_patterns(["archive/**", "2025/**"], "ignore_path")

    corpus = [
        CorpusPaper(
            title="Active Paper",
            abstract="",
            added_date=datetime(2026, 1, 1),
            paths=["2026/survey/topic-a"],
        ),
        CorpusPaper(
            title="Archived Paper",
            abstract="",
            added_date=datetime(2026, 1, 2),
            paths=["archive/misc"],
        ),
        CorpusPaper(
            title="Old Paper",
            abstract="",
            added_date=datetime(2026, 1, 3),
            paths=["2025/other/topic"],
        ),
    ]

    filtered = executor.filter_corpus(corpus)

    assert [paper.title for paper in filtered] == ["Active Paper"]


def test_filter_corpus_ignore_path_takes_precedence_over_include_path():
    """Papers matching both include_path and ignore_path should be excluded."""
    executor = Executor.__new__(Executor)
    executor.include_path_patterns = normalize_path_patterns(["2026/**"], "include_path")
    executor.ignore_path_patterns = normalize_path_patterns(["2026/ignore/**"], "ignore_path")

    corpus = [
        CorpusPaper(
            title="Included Paper",
            abstract="",
            added_date=datetime(2026, 1, 1),
            paths=["2026/survey/topic-a"],
        ),
        CorpusPaper(
            title="Ignored Paper",
            abstract="",
            added_date=datetime(2026, 1, 2),
            paths=["2026/ignore/topic-b"],
        ),
    ]

    filtered = executor.filter_corpus(corpus)

    assert [paper.title for paper in filtered] == ["Included Paper"]


def test_filter_corpus_no_filters_returns_all():
    executor = Executor.__new__(Executor)
    executor.include_path_patterns = None
    executor.ignore_path_patterns = None

    corpus = [
        CorpusPaper(title="Paper A", abstract="", added_date=datetime(2026, 1, 1), paths=["foo"]),
        CorpusPaper(title="Paper B", abstract="", added_date=datetime(2026, 1, 2), paths=["bar"]),
    ]

    filtered = executor.filter_corpus(corpus)

    assert filtered == corpus
