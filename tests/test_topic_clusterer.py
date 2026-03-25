import json
from types import SimpleNamespace

import pytest
from openai import OpenAIError

from zotero_arxiv_daily.protocol import Paper
from zotero_arxiv_daily.topic_clusterer import TopicClusterer


@pytest.fixture
def llm_config() -> dict:
    return {"generation_kwargs": {}}


class FakeChatClient:
    def __init__(self, responses):
        if isinstance(responses, (str, Exception)):
            responses = [responses]
        self.responses = list(responses)
        self.calls = 0
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, *args, **kwargs):
        self.calls += 1
        self.requests.append(kwargs)
        index = min(self.calls - 1, len(self.responses) - 1)
        response = self.responses[index]
        if isinstance(response, Exception):
            raise response
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=response)
                )
            ]
        )


def make_paper(title: str, abstract: str, url_suffix: str, tldr: str | None) -> Paper:
    return Paper(
        source="arxiv",
        title=title,
        authors=[title],
        abstract=abstract,
        url=f"https://example.com/{url_suffix}",
        pdf_url=f"https://example.com/{url_suffix}.pdf",
        tldr=tldr,
        score=9.0,
    )


def make_six_papers() -> list[Paper]:
    return [
        make_paper("A", "vision abstract", "a", "vision topic"),
        make_paper("B", "reasoning abstract", "b", "reasoning topic"),
        make_paper("C", "vision abstract fallback", "c", None),
        make_paper("D", "agents abstract", "d", "agent topic"),
        make_paper("E", "agents abstract 2", "e", "agent topic 2"),
        make_paper("F", "reasoning abstract 2", "f", "reasoning topic 2"),
    ]


def make_three_papers() -> list[Paper]:
    return make_six_papers()[:3]


def test_cluster_papers_returns_sorted_groups_from_valid_json(llm_config):
    papers = make_six_papers()
    client = FakeChatClient(
        """{
  \"groups\": [
    {\"label\": \"Agents\", \"summary\": \"Agent systems papers.\", \"paper_indices\": [4, 3]},
    {\"label\": \"Vision\", \"summary\": \"Vision papers.\", \"paper_indices\": [2, 0]},
    {\"label\": \"Reasoning\", \"summary\": \"Reasoning papers.\", \"paper_indices\": [5, 1]}
  ]
}"""
    )

    groups = TopicClusterer(client, llm_config).cluster_papers(papers)

    assert [group.label for group in groups] == ["Vision", "Reasoning", "Agents"]
    assert groups[0].summary == "Vision papers."
    assert [paper.title for paper in groups[0].papers] == ["A", "C"]
    assert [paper.title for paper in groups[1].papers] == ["B", "F"]
    assert [paper.title for paper in groups[2].papers] == ["D", "E"]

    payload = json.loads(client.requests[0]["messages"][1]["content"])
    assert payload["papers"][0] == {
        "index": 0,
        "title": "A",
        "summary_text": "vision topic",
    }
    assert payload["papers"][2]["summary_text"] == "vision abstract fallback"


def test_cluster_papers_falls_back_when_indices_are_invalid(llm_config):
    papers = make_six_papers()
    client = FakeChatClient(
        """{
  \"groups\": [
    {\"label\": \"Vision\", \"summary\": \"Vision papers.\", \"paper_indices\": [0, 0]},
    {\"label\": \"Reasoning\", \"summary\": \"Reasoning papers.\", \"paper_indices\": [1, 2]},
    {\"label\": \"Agents\", \"summary\": \"Agent papers.\", \"paper_indices\": [3, 4]}
  ]
}"""
    )

    groups = TopicClusterer(client, llm_config).cluster_papers(papers)

    assert client.calls == 2
    assert len(groups) == 1
    assert groups[0].label == "Relevant papers today"
    assert groups[0].summary is None
    assert groups[0].papers == papers


def test_cluster_papers_skips_llm_when_too_few_papers(llm_config):
    papers = make_three_papers()
    client = FakeChatClient('{"groups": []}')

    groups = TopicClusterer(client, llm_config).cluster_papers(papers)

    assert len(groups) == 1
    assert groups[0].label == "Relevant papers today"
    assert groups[0].summary is None
    assert groups[0].papers == papers
    assert client.calls == 0


def test_cluster_papers_retries_once_on_invalid_then_succeeds(llm_config):
    papers = make_six_papers()
    client = FakeChatClient(
        [
            "not valid json",
            """{
  \"groups\": [
    {\"label\": \"Agents\", \"summary\": \"Agent systems papers.\", \"paper_indices\": [3, 4]},
    {\"label\": \"Vision\", \"summary\": \"Vision papers.\", \"paper_indices\": [0, 2]},
    {\"label\": \"Reasoning\", \"summary\": \"Reasoning papers.\", \"paper_indices\": [1, 5]}
  ]
}""",
        ]
    )

    groups = TopicClusterer(client, llm_config).cluster_papers(papers)

    assert client.calls == 2
    assert [group.label for group in groups] == ["Vision", "Reasoning", "Agents"]
    assert "Return JSON only" in client.requests[1]["messages"][0]["content"]
    assert client.requests[1]["messages"][0]["content"] != client.requests[0]["messages"][0]["content"]


def test_cluster_papers_missing_summaries_trigger_retry_then_fallback(llm_config):
    papers = make_six_papers()
    missing_summary = """{
  \"groups\": [
    {\"label\": \"Vision\", \"summary\": \"\", \"paper_indices\": [0, 2]},
    {\"label\": \"Reasoning\", \"summary\": \"Reasoning papers.\", \"paper_indices\": [1, 5]},
    {\"label\": \"Agents\", \"summary\": \"Agent papers.\", \"paper_indices\": [3, 4]}
  ]
}"""
    client = FakeChatClient([missing_summary, missing_summary])

    groups = TopicClusterer(client, llm_config).cluster_papers(papers)

    assert client.calls == 2
    assert len(groups) == 1
    assert groups[0].label == "Relevant papers today"
    assert groups[0].summary is None
    assert groups[0].papers == papers


def test_cluster_papers_rejects_generic_group_labels(llm_config):
    papers = make_six_papers()
    generic_labels = """{
  \"groups\": [
    {\"label\": \"Group 1\", \"summary\": \"Vision papers.\", \"paper_indices\": [0, 2]},
    {\"label\": \"Reasoning\", \"summary\": \"Reasoning papers.\", \"paper_indices\": [1, 5]},
    {\"label\": \"Agents\", \"summary\": \"Agent papers.\", \"paper_indices\": [3, 4]}
  ]
}"""
    client = FakeChatClient([generic_labels, generic_labels])

    groups = TopicClusterer(client, llm_config).cluster_papers(papers)

    assert client.calls == 2
    assert len(groups) == 1
    assert groups[0].label == "Relevant papers today"
    assert groups[0].summary is None
    assert groups[0].papers == papers


def test_cluster_papers_rejects_two_groups_for_six_papers(llm_config):
    papers = make_six_papers()
    two_groups = """{
  \"groups\": [
    {\"label\": \"Vision\", \"summary\": \"Vision papers.\", \"paper_indices\": [0, 1, 2]},
    {\"label\": \"Agents\", \"summary\": \"Agent and reasoning papers.\", \"paper_indices\": [3, 4, 5]}
  ]
}"""
    client = FakeChatClient([two_groups, two_groups])

    groups = TopicClusterer(client, llm_config).cluster_papers(papers)

    assert client.calls == 2
    assert len(groups) == 1
    assert groups[0].label == "Relevant papers today"
    assert groups[0].summary is None
    assert groups[0].papers == papers


def test_cluster_papers_rejects_six_groups_for_six_papers(llm_config):
    papers = make_six_papers()
    six_groups = """{
  \"groups\": [
    {\"label\": \"Vision\", \"summary\": \"Vision paper.\", \"paper_indices\": [0]},
    {\"label\": \"Reasoning\", \"summary\": \"Reasoning paper.\", \"paper_indices\": [1]},
    {\"label\": \"Perception\", \"summary\": \"Perception paper.\", \"paper_indices\": [2]},
    {\"label\": \"Agents\", \"summary\": \"Agent paper.\", \"paper_indices\": [3]},
    {\"label\": \"Planning\", \"summary\": \"Planning paper.\", \"paper_indices\": [4]},
    {\"label\": \"Evaluation\", \"summary\": \"Evaluation paper.\", \"paper_indices\": [5]}
  ]
}"""
    client = FakeChatClient([six_groups, six_groups])

    groups = TopicClusterer(client, llm_config).cluster_papers(papers)

    assert client.calls == 2
    assert len(groups) == 1
    assert groups[0].label == "Relevant papers today"
    assert groups[0].summary is None
    assert groups[0].papers == papers


class FakeTransportError(OpenAIError):
    pass


def test_cluster_papers_requests_json_output_mode(llm_config):
    papers = make_six_papers()
    client = FakeChatClient(
        """{
  \"groups\": [
    {\"label\": \"Agents\", \"summary\": \"Agent systems papers.\", \"paper_indices\": [3, 4]},
    {\"label\": \"Vision\", \"summary\": \"Vision papers.\", \"paper_indices\": [0, 2]},
    {\"label\": \"Reasoning\", \"summary\": \"Reasoning papers.\", \"paper_indices\": [1, 5]}
  ]
}"""
    )

    TopicClusterer(client, llm_config).cluster_papers(papers)

    assert client.requests[0]["response_format"] == {"type": "json_object"}


def test_cluster_papers_does_not_retry_on_transport_failure(llm_config):
    papers = make_six_papers()
    client = FakeChatClient([FakeTransportError("gateway down")])

    groups = TopicClusterer(client, llm_config).cluster_papers(papers)

    assert client.calls == 1
    assert len(groups) == 1
    assert groups[0].label == "Relevant papers today"
    assert groups[0].summary is None
    assert groups[0].papers == papers
