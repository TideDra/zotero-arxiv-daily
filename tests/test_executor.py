"""Tests for zotero_arxiv_daily.executor: normalize_path_patterns, filter_corpus, fetch_zotero_corpus, E2E."""

from datetime import datetime

import pytest
from omegaconf import OmegaConf

from zotero_arxiv_daily.executor import Executor, normalize_path_patterns
from zotero_arxiv_daily.protocol import CorpusPaper, Paper


# ---------------------------------------------------------------------------
# normalize_path_patterns — migrated from test_include_path.py
# ---------------------------------------------------------------------------


def test_normalize_path_patterns_rejects_single_string_for_include_path():
    with pytest.raises(TypeError, match="config.zotero.include_path must be a list"):
        normalize_path_patterns("2026/survey/**", "include_path")


def test_normalize_path_patterns_accepts_list_config_for_include_path():
    include_path = OmegaConf.create(["2026/survey/**", "2026/reading-group/**"])
    assert normalize_path_patterns(include_path, "include_path") == [
        "2026/survey/**",
        "2026/reading-group/**",
    ]


def test_normalize_path_patterns_rejects_single_string_for_ignore_path():
    with pytest.raises(TypeError, match="config.zotero.ignore_path must be a list"):
        normalize_path_patterns("archive/**", "ignore_path")


def test_normalize_path_patterns_accepts_list_config_for_ignore_path():
    ignore_path = OmegaConf.create(["archive/**", "2025/**"])
    assert normalize_path_patterns(ignore_path, "ignore_path") == ["archive/**", "2025/**"]


def test_normalize_path_patterns_accepts_empty_list():
    assert normalize_path_patterns([], "ignore_path") == []


def test_normalize_path_patterns_accepts_none():
    assert normalize_path_patterns(None, "include_path") is None


# ---------------------------------------------------------------------------
# filter_corpus — migrated from test_include_path.py
# ---------------------------------------------------------------------------


def _make_executor(include_patterns=None, ignore_patterns=None):
    executor = Executor.__new__(Executor)
    executor.include_path_patterns = normalize_path_patterns(include_patterns, "include_path") if include_patterns else None
    executor.ignore_path_patterns = normalize_path_patterns(ignore_patterns, "ignore_path") if ignore_patterns else None
    return executor


def test_filter_corpus_matches_any_path_against_any_pattern():
    executor = _make_executor(include_patterns=["2026/survey/**", "2026/reading-group/**"])
    corpus = [
        CorpusPaper(title="Survey Paper", abstract="", added_date=datetime(2026, 1, 1), paths=["2026/survey/topic-a", "archive/misc"]),
        CorpusPaper(title="Reading Group Paper", abstract="", added_date=datetime(2026, 1, 2), paths=["notes/inbox", "2026/reading-group/week-1"]),
        CorpusPaper(title="Excluded Paper", abstract="", added_date=datetime(2026, 1, 3), paths=["2025/other/topic"]),
    ]
    filtered = executor.filter_corpus(corpus)
    assert [p.title for p in filtered] == ["Survey Paper", "Reading Group Paper"]


def test_filter_corpus_excludes_papers_matching_ignore_path():
    executor = _make_executor(ignore_patterns=["archive/**", "2025/**"])
    corpus = [
        CorpusPaper(title="Active Paper", abstract="", added_date=datetime(2026, 1, 1), paths=["2026/survey/topic-a"]),
        CorpusPaper(title="Archived Paper", abstract="", added_date=datetime(2026, 1, 2), paths=["archive/misc"]),
        CorpusPaper(title="Old Paper", abstract="", added_date=datetime(2026, 1, 3), paths=["2025/other/topic"]),
    ]
    filtered = executor.filter_corpus(corpus)
    assert [p.title for p in filtered] == ["Active Paper"]


def test_filter_corpus_ignore_path_takes_precedence_over_include_path():
    executor = _make_executor(include_patterns=["2026/**"], ignore_patterns=["2026/ignore/**"])
    corpus = [
        CorpusPaper(title="Included Paper", abstract="", added_date=datetime(2026, 1, 1), paths=["2026/survey/topic-a"]),
        CorpusPaper(title="Ignored Paper", abstract="", added_date=datetime(2026, 1, 2), paths=["2026/ignore/topic-b"]),
    ]
    filtered = executor.filter_corpus(corpus)
    assert [p.title for p in filtered] == ["Included Paper"]


def test_filter_corpus_no_filters_returns_all():
    executor = _make_executor()
    corpus = [
        CorpusPaper(title="Paper A", abstract="", added_date=datetime(2026, 1, 1), paths=["foo"]),
        CorpusPaper(title="Paper B", abstract="", added_date=datetime(2026, 1, 2), paths=["bar"]),
    ]
    filtered = executor.filter_corpus(corpus)
    assert filtered == corpus


# ---------------------------------------------------------------------------
# fetch_zotero_corpus
# ---------------------------------------------------------------------------


def test_fetch_zotero_corpus(config, monkeypatch):
    from tests.canned_responses import make_stub_zotero_client

    stub_zot = make_stub_zotero_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.zotero.Zotero", lambda *a, **kw: stub_zot)

    executor = Executor.__new__(Executor)
    executor.config = config
    corpus = executor.fetch_zotero_corpus()

    assert len(corpus) == 2
    assert corpus[0].title == "Stub Paper 1"
    assert "survey/topic-a" in corpus[0].paths[0]


def test_fetch_zotero_corpus_paper_with_zero_collections(config, monkeypatch):
    from tests.canned_responses import make_stub_zotero_client

    items = [
        {
            "data": {
                "title": "No Collection Paper",
                "abstractNote": "Abstract.",
                "dateAdded": "2026-03-01T00:00:00Z",
                "collections": [],
            }
        }
    ]
    stub_zot = make_stub_zotero_client(items=items)
    monkeypatch.setattr("zotero_arxiv_daily.executor.zotero.Zotero", lambda *a, **kw: stub_zot)

    executor = Executor.__new__(Executor)
    executor.config = config
    corpus = executor.fetch_zotero_corpus()

    assert len(corpus) == 1
    assert corpus[0].paths == []


# ---------------------------------------------------------------------------
# E2E: Executor.run()
# ---------------------------------------------------------------------------


def test_run_end_to_end(config, monkeypatch):
    """Full pipeline: Zotero fetch -> filter -> retrieve -> rerank -> TLDR -> email."""
    import smtplib

    from omegaconf import open_dict

    from tests.canned_responses import (
        make_sample_corpus,
        make_sample_paper,
        make_stub_openai_client,
        make_stub_smtp,
        make_stub_zotero_client,
    )

    # Config: source=["arxiv"], reranker="api", send_empty=false
    with open_dict(config):
        config.executor.source = ["arxiv"]
        config.executor.reranker = "api"
        config.executor.send_empty = False

    # 1. Stub pyzotero
    stub_zot = make_stub_zotero_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.zotero.Zotero", lambda *a, **kw: stub_zot)

    # 2. Stub OpenAI (for reranker + TLDR/affiliations)
    stub_client = make_stub_openai_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.OpenAI", lambda **kw: stub_client)
    monkeypatch.setattr("zotero_arxiv_daily.reranker.api.OpenAI", lambda **kw: stub_client)
    retrieved = [
        make_sample_paper(title="E2E Paper 1", score=None),
        make_sample_paper(title="E2E Paper 2", score=None),
    ]

    # Import to register the arxiv retriever
    import zotero_arxiv_daily.retriever.arxiv_retriever  # noqa: F401

    from zotero_arxiv_daily.retriever.base import registered_retrievers

    monkeypatch.setattr(
        registered_retrievers["arxiv"],
        "retrieve_papers",
        lambda self: retrieved,
    )

    # 4. Stub SMTP
    sent = []
    monkeypatch.setattr(smtplib, "SMTP", make_stub_smtp(sent))

    # 5. Stub sleep (reranker/retriever)
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)

    # 6. Run
    executor = Executor(config)
    executor.run()

    # Assertions
    assert len(sent) == 1, "Email should have been sent"
    _, _, email_body = sent[0]
    assert "text/html" in email_body


def test_run_no_papers_send_empty_false(config, monkeypatch, tmp_path):
    """When no papers are found, skip email but still publish a valid empty feed."""
    import smtplib
    import xml.etree.ElementTree as ET

    from omegaconf import open_dict

    from tests.canned_responses import make_stub_openai_client, make_stub_smtp, make_stub_zotero_client

    with open_dict(config):
        config.executor.source = ["arxiv"]
        config.executor.reranker = "api"
        config.executor.send_empty = False
        config.output.atom.enabled = True
        config.output.atom.path = str(tmp_path / "index.xml")

    stub_zot = make_stub_zotero_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.zotero.Zotero", lambda *a, **kw: stub_zot)

    stub_client = make_stub_openai_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.OpenAI", lambda **kw: stub_client)
    monkeypatch.setattr("zotero_arxiv_daily.reranker.api.OpenAI", lambda **kw: stub_client)

    import zotero_arxiv_daily.retriever.arxiv_retriever  # noqa: F401

    from zotero_arxiv_daily.retriever.base import registered_retrievers

    monkeypatch.setattr(registered_retrievers["arxiv"], "retrieve_papers", lambda self: [])

    sent = []
    monkeypatch.setattr(smtplib, "SMTP", make_stub_smtp(sent))
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)

    executor = Executor(config)
    executor.run()

    assert len(sent) == 0, "No email should be sent when no papers and send_empty=false"
    feed = ET.parse(tmp_path / "index.xml")
    assert feed.findall("{http://www.w3.org/2005/Atom}entry") == []


def test_run_no_papers_send_empty_true(config, monkeypatch):
    """When no papers are found and send_empty=true, empty email is sent."""
    import smtplib

    from omegaconf import open_dict

    from tests.canned_responses import make_stub_openai_client, make_stub_smtp, make_stub_zotero_client

    with open_dict(config):
        config.executor.source = ["arxiv"]
        config.executor.reranker = "api"
        config.executor.send_empty = True

    stub_zot = make_stub_zotero_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.zotero.Zotero", lambda *a, **kw: stub_zot)

    stub_client = make_stub_openai_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.OpenAI", lambda **kw: stub_client)
    monkeypatch.setattr("zotero_arxiv_daily.reranker.api.OpenAI", lambda **kw: stub_client)

    import zotero_arxiv_daily.retriever.arxiv_retriever  # noqa: F401

    from zotero_arxiv_daily.retriever.base import registered_retrievers

    monkeypatch.setattr(registered_retrievers["arxiv"], "retrieve_papers", lambda self: [])

    sent = []
    monkeypatch.setattr(smtplib, "SMTP", make_stub_smtp(sent))
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)

    executor = Executor(config)
    executor.run()

    assert len(sent) == 1, "Email should be sent even with no papers when send_empty=true"
    _, _, body = sent[0]
    assert "text/html" in body


@pytest.mark.parametrize("source", ["scix", "ads"])
def test_scix_and_ads_alias_reject_api_reranker(config, source):
    from omegaconf import open_dict

    with open_dict(config):
        config.executor.source = [source]
        config.executor.reranker = "api"
        source_config = getattr(config.source, source)
        source_config.api_token = "scix-secret"
        source_config.query = "database:astronomy"
    with pytest.raises(ValueError, match="reranker=local"):
        Executor(config)


def test_scix_and_ads_alias_cannot_be_enabled_together(config):
    from omegaconf import open_dict

    with open_dict(config):
        config.executor.source = ["scix", "ads"]
        config.executor.reranker = "local"
    with pytest.raises(ValueError, match="only one"):
        Executor(config)


def test_scix_only_record_stays_out_of_remote_and_public_feed(config, monkeypatch, tmp_path):
    """Exercise the full SciX privacy boundary, delivery state, email, and Atom output."""
    import json
    import smtplib
    from email import message_from_string

    from omegaconf import open_dict

    from tests.canned_responses import make_stub_smtp, make_stub_zotero_client
    from zotero_arxiv_daily.reranker.base import registered_rerankers
    from zotero_arxiv_daily.retriever.base import registered_retrievers

    state_path = tmp_path / "state" / "ads.json"
    feed_path = tmp_path / "public" / "index.xml"
    with open_dict(config):
        config.executor.source = ["scix"]
        config.executor.reranker = "local"
        config.source.scix.api_token = "scix-secret"
        config.source.scix.query = "database:astronomy"
        config.source.scix.state_path = str(state_path)
        config.output.atom.enabled = True
        config.output.atom.path = str(feed_path)

    sentinel = "PRIVATE SCIX ABSTRACT SENTINEL"
    retrieved = [
        Paper(
            source="scix",
            title="SciX-only candidate",
            authors=["A. Astronomer"],
            abstract=sentinel,
            url="https://scixplorer.org/abs/2026TEST....1A/abstract",
            external_ids={"ads": "2026TEST....1A"},
            content_source="scix",
            remote_processing_allowed=False,
        )
    ]
    monkeypatch.setattr(
        registered_retrievers["scix"],
        "retrieve_papers",
        lambda self: retrieved,
    )

    def rank_locally(self, candidates, corpus):
        candidates[0].score = 9.0
        return candidates

    monkeypatch.setattr(registered_rerankers["local"], "rerank", rank_locally)
    monkeypatch.setattr(
        "zotero_arxiv_daily.executor.zotero.Zotero",
        lambda *args, **kwargs: make_stub_zotero_client(),
    )

    remote_calls = []

    def forbidden_remote_call(**kwargs):
        remote_calls.append(kwargs)
        raise AssertionError("SciX-only content reached the remote client")

    remote_client = type(
        "RemoteClient",
        (),
        {
            "chat": type(
                "Chat",
                (),
                {"completions": type("Completions", (), {"create": staticmethod(forbidden_remote_call)})()},
            )()
        },
    )()
    monkeypatch.setattr("zotero_arxiv_daily.executor.OpenAI", lambda **kwargs: remote_client)

    sent = []
    monkeypatch.setattr(smtplib, "SMTP", make_stub_smtp(sent))

    result = Executor(config).run()

    assert result == retrieved
    assert remote_calls == []
    message = message_from_string(sent[0][2])
    html = message.get_payload(decode=True).decode("utf-8")
    assert "SciX-only candidate" in html
    assert sentinel not in html
    feed_xml = feed_path.read_text(encoding="utf-8")
    assert sentinel not in feed_xml
    assert "SciX-only candidate" not in feed_xml
    assert "2026TEST....1A" not in feed_xml
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["seen"].keys() == {"ads:2026TEST....1A"}
    assert sentinel not in json.dumps(state)


def test_scix_state_marks_only_papers_actually_delivered(config, monkeypatch, tmp_path):
    import json
    import smtplib

    from omegaconf import open_dict

    from tests.canned_responses import make_stub_smtp, make_stub_zotero_client
    from zotero_arxiv_daily.reranker.base import registered_rerankers
    from zotero_arxiv_daily.retriever.base import registered_retrievers

    state_path = tmp_path / "state" / "ads.json"
    with open_dict(config):
        config.executor.source = ["scix"]
        config.executor.reranker = "local"
        config.executor.max_paper_num = 1
        config.source.scix.api_token = "scix-secret"
        config.source.scix.query = "database:astronomy"
        config.source.scix.state_path = str(state_path)
        config.output.atom.enabled = False

    retrieved = [
        Paper(
            source="scix",
            title=f"Candidate {index}",
            authors=["A. Astronomer"],
            abstract=f"Private abstract {index}",
            url=f"https://scixplorer.org/abs/2026TEST....{index}A/abstract",
            external_ids={"ads": f"2026TEST....{index}A"},
            content_source="scix",
            remote_processing_allowed=False,
        )
        for index in (1, 2)
    ]
    monkeypatch.setattr(registered_retrievers["scix"], "retrieve_papers", lambda self: retrieved)
    monkeypatch.setattr(registered_rerankers["local"], "rerank", lambda self, candidates, corpus: candidates)
    monkeypatch.setattr(
        "zotero_arxiv_daily.executor.zotero.Zotero",
        lambda *args, **kwargs: make_stub_zotero_client(),
    )
    monkeypatch.setattr(
        "zotero_arxiv_daily.executor.OpenAI",
        lambda **kwargs: type("NoRemoteClient", (), {})(),
    )
    sent = []
    monkeypatch.setattr(smtplib, "SMTP", make_stub_smtp(sent))

    result = Executor(config).run()

    assert [paper.title for paper in result] == ["Candidate 1"]
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["seen"].keys() == {"ads:2026TEST....1A"}
