from copy import deepcopy
from datetime import datetime
from types import SimpleNamespace

import pytest

import zotero_arxiv_daily.executor as executor_module
from zotero_arxiv_daily.protocol import CorpusPaper, Paper
from zotero_arxiv_daily.topic_clusterer import PaperGroup, TopicClusterer


class DummyOpenAI:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


def make_executor_config(config, *, send_empty: bool, max_paper_num: int = 10):
    executor_config = deepcopy(config)
    executor_config.executor.source = ["arxiv"]
    executor_config.executor.reranker = "test-reranker"
    executor_config.executor.max_paper_num = max_paper_num
    executor_config.executor.send_empty = send_empty
    return executor_config


def make_corpus() -> list[CorpusPaper]:
    return [
        CorpusPaper(
            title="Existing corpus paper",
            abstract="A corpus paper used for executor tests.",
            added_date=datetime(2025, 1, 1),
            paths=["ml/vision"],
        )
    ]


def stub_generation_methods(paper: Paper, index: int) -> Paper:
    def generate_tldr(_openai_client, _llm_params, paper=paper, index=index):
        paper.tldr = f"TLDR {index}"
        return paper.tldr

    def generate_affiliations(_openai_client, _llm_params, paper=paper, index=index):
        paper.affiliations = [f"Affiliation {index}"]
        return paper.affiliations

    paper.generate_tldr = generate_tldr
    paper.generate_affiliations = generate_affiliations
    return paper


def make_papers(count: int) -> list[Paper]:
    return [
        stub_generation_methods(
            Paper(
                source="arxiv",
                title=f"Paper {index}",
                authors=[f"Author {index}"],
                abstract=f"Abstract {index}",
                url=f"https://example.com/paper-{index}",
                pdf_url=f"https://example.com/paper-{index}.pdf",
                full_text=f"Full text {index}",
                score=10 - index,
            ),
            index,
        )
        for index in range(count)
    ]


def make_render_email_recorder(calls: list, html: str = "<html>rendered email</html>"):
    def fake_render_email(groups):
        calls.append(groups)
        return html

    return fake_render_email


def make_send_email_recorder(calls: list):
    def fake_send_email(config, html):
        calls.append((config, html))

    return fake_send_email


def build_executor(monkeypatch: pytest.MonkeyPatch, config, *, retrieved_papers: list[Paper], reranked_papers: list[Paper]):
    class FakeRetriever:
        def __init__(self, _config):
            pass

        def retrieve_papers(self):
            return list(retrieved_papers)

    class FakeReranker:
        def __init__(self, _config):
            pass

        def rerank(self, papers, corpus):
            return list(reranked_papers)

    monkeypatch.setattr(executor_module, "OpenAI", DummyOpenAI)
    monkeypatch.setattr(executor_module, "get_retriever_cls", lambda _source: FakeRetriever)
    monkeypatch.setattr(executor_module, "get_reranker_cls", lambda _name: FakeReranker)

    executor = executor_module.Executor(config)
    monkeypatch.setattr(executor, "fetch_zotero_corpus", lambda: make_corpus())
    monkeypatch.setattr(executor, "filter_corpus", lambda corpus: corpus)
    return executor


def make_topic_clusterer_class(return_value, observed: dict):
    class FakeTopicClusterer:
        def __init__(self, openai_client, llm_params):
            observed["init_args"] = (openai_client, llm_params)

        def cluster_papers(self, papers):
            observed.setdefault("cluster_calls", []).append(list(papers))
            return return_value

    return FakeTopicClusterer


def make_forwarding_topic_clusterer(observed: dict):
    class ForwardingTopicClusterer:
        def __init__(self, openai_client, llm_params):
            self.delegate = TopicClusterer(openai_client, llm_params)

        def cluster_papers(self, papers):
            observed.setdefault("cluster_calls", []).append(list(papers))
            return self.delegate.cluster_papers(papers)

    return ForwardingTopicClusterer


def test_executor_initializes_topic_clusterer_with_openai_client_and_llm_config(
    config,
    monkeypatch: pytest.MonkeyPatch,
):
    test_config = make_executor_config(config, send_empty=False)
    cluster_observed = {}
    fake_topic_clusterer_cls = make_topic_clusterer_class([], cluster_observed)
    monkeypatch.setattr(executor_module, "TopicClusterer", fake_topic_clusterer_cls, raising=False)

    executor = build_executor(
        monkeypatch,
        test_config,
        retrieved_papers=[],
        reranked_papers=[],
    )

    assert isinstance(executor.openai_client, DummyOpenAI)
    assert isinstance(executor.topic_clusterer, fake_topic_clusterer_cls)
    assert cluster_observed["init_args"] == (executor.openai_client, test_config.llm)


def test_executor_passes_clustered_groups_to_render_email(config, monkeypatch: pytest.MonkeyPatch):
    test_config = make_executor_config(config, send_empty=False, max_paper_num=2)
    reranked_papers = make_papers(3)
    executor = build_executor(
        monkeypatch,
        test_config,
        retrieved_papers=reranked_papers,
        reranked_papers=reranked_papers,
    )

    expected_groups = [
        PaperGroup(
            label="Vision",
            summary="Papers about vision.",
            papers=reranked_papers[:2],
        )
    ]
    cluster_observed = {"cluster_calls": []}
    render_calls = []
    send_calls = []

    executor.topic_clusterer = SimpleNamespace(
        cluster_papers=lambda papers: cluster_observed.setdefault("cluster_calls", []).append(list(papers)) or expected_groups
    )
    monkeypatch.setattr(executor_module, "render_email", make_render_email_recorder(render_calls))
    monkeypatch.setattr(executor_module, "send_email", make_send_email_recorder(send_calls))

    executor.run()

    assert len(cluster_observed["cluster_calls"]) == 1
    clustered_papers = cluster_observed["cluster_calls"][0]
    assert clustered_papers == reranked_papers[:2]
    assert [paper.tldr for paper in clustered_papers] == ["TLDR 0", "TLDR 1"]
    assert [paper.affiliations for paper in clustered_papers] == [["Affiliation 0"], ["Affiliation 1"]]
    assert render_calls == [expected_groups]
    assert send_calls == [(test_config, "<html>rendered email</html>")]


def test_executor_passes_fallback_group_to_render_email(config, monkeypatch: pytest.MonkeyPatch):
    test_config = make_executor_config(config, send_empty=False, max_paper_num=3)
    reranked_papers = make_papers(3)
    executor = build_executor(
        monkeypatch,
        test_config,
        retrieved_papers=reranked_papers,
        reranked_papers=reranked_papers,
    )

    cluster_observed = {"cluster_calls": []}
    render_calls = []
    send_calls = []

    executor.topic_clusterer = make_forwarding_topic_clusterer(cluster_observed)(executor.openai_client, test_config.llm)
    monkeypatch.setattr(executor_module, "render_email", make_render_email_recorder(render_calls))
    monkeypatch.setattr(executor_module, "send_email", make_send_email_recorder(send_calls))

    executor.run()

    assert len(cluster_observed["cluster_calls"]) == 1
    assert cluster_observed["cluster_calls"][0] == reranked_papers
    assert len(render_calls) == 1
    grouped_papers = render_calls[0]
    assert len(grouped_papers) == 1
    assert grouped_papers[0].label == "Relevant papers today"
    assert grouped_papers[0].summary is None
    assert grouped_papers[0].papers == reranked_papers
    assert send_calls == [(test_config, "<html>rendered email</html>")]


def test_executor_skips_email_when_no_papers_and_send_empty_disabled(config, monkeypatch: pytest.MonkeyPatch):
    test_config = make_executor_config(config, send_empty=False)
    executor = build_executor(
        monkeypatch,
        test_config,
        retrieved_papers=[],
        reranked_papers=[],
    )

    cluster_observed = {"cluster_calls": []}
    render_calls = []
    send_calls = []

    executor.topic_clusterer = make_topic_clusterer_class([], cluster_observed)(executor.openai_client, test_config.llm)
    monkeypatch.setattr(executor_module, "render_email", make_render_email_recorder(render_calls))
    monkeypatch.setattr(executor_module, "send_email", make_send_email_recorder(send_calls))

    executor.run()

    assert cluster_observed["cluster_calls"] == []
    assert render_calls == []
    assert send_calls == []


def test_executor_sends_empty_email_without_clustering_when_no_papers_and_send_empty_enabled(
    config,
    monkeypatch: pytest.MonkeyPatch,
):
    test_config = make_executor_config(config, send_empty=True)
    executor = build_executor(
        monkeypatch,
        test_config,
        retrieved_papers=[],
        reranked_papers=[],
    )

    cluster_observed = {"cluster_calls": []}
    render_calls = []
    send_calls = []

    executor.topic_clusterer = make_topic_clusterer_class([], cluster_observed)(executor.openai_client, test_config.llm)
    monkeypatch.setattr(executor_module, "render_email", make_render_email_recorder(render_calls))
    monkeypatch.setattr(executor_module, "send_email", make_send_email_recorder(send_calls))

    executor.run()

    assert cluster_observed["cluster_calls"] == []
    assert render_calls == [[]]
    assert send_calls == [(test_config, "<html>rendered email</html>")]


def test_executor_skips_email_when_reranking_returns_no_selected_papers_and_send_empty_disabled(
    config,
    monkeypatch: pytest.MonkeyPatch,
):
    test_config = make_executor_config(config, send_empty=False)
    retrieved_papers = make_papers(2)
    executor = build_executor(
        monkeypatch,
        test_config,
        retrieved_papers=retrieved_papers,
        reranked_papers=[],
    )

    cluster_observed = {"cluster_calls": []}
    render_calls = []
    send_calls = []

    executor.topic_clusterer = make_topic_clusterer_class([], cluster_observed)(executor.openai_client, test_config.llm)
    monkeypatch.setattr(executor_module, "render_email", make_render_email_recorder(render_calls))
    monkeypatch.setattr(executor_module, "send_email", make_send_email_recorder(send_calls))

    executor.run()

    assert cluster_observed["cluster_calls"] == []
    assert render_calls == []
    assert send_calls == []


def test_executor_sends_empty_email_when_reranking_returns_no_selected_papers_and_send_empty_enabled(
    config,
    monkeypatch: pytest.MonkeyPatch,
):
    test_config = make_executor_config(config, send_empty=True)
    retrieved_papers = make_papers(2)
    executor = build_executor(
        monkeypatch,
        test_config,
        retrieved_papers=retrieved_papers,
        reranked_papers=[],
    )

    cluster_observed = {"cluster_calls": []}
    render_calls = []
    send_calls = []

    executor.topic_clusterer = make_topic_clusterer_class([], cluster_observed)(executor.openai_client, test_config.llm)
    monkeypatch.setattr(executor_module, "render_email", make_render_email_recorder(render_calls))
    monkeypatch.setattr(executor_module, "send_email", make_send_email_recorder(send_calls))

    executor.run()

    assert cluster_observed["cluster_calls"] == []
    assert render_calls == [[]]
    assert send_calls == [(test_config, "<html>rendered email</html>")]
