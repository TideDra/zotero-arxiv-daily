"""Tests for ApiReranker — uses stub OpenAI client via monkeypatch."""

from zotero_arxiv_daily.reranker.api import ApiReranker


def test_api_reranker_similarity_shape(config, patch_openai):
    reranker = ApiReranker(config)
    score = reranker.get_similarity_score(["hello", "world"], ["ping"])
    assert score.shape == (2, 1)


def test_api_reranker_batching(config, patch_openai):
    reranker = ApiReranker(config)
    s1 = [f"text {i}" for i in range(5)]
    s2 = [f"corpus {i}" for i in range(3)]
    score = reranker.get_similarity_score(s1, s2)
    assert score.shape == (5, 3)


def test_api_reranker_litellm(config, monkeypatch):
    """Embeddings route through litellm.embedding when reranker.api.use_litellm."""
    from types import SimpleNamespace

    config.reranker.api.use_litellm = True
    config.reranker.api.model = "voyage/voyage-3"

    def fake_embedding(**kwargs):
        assert kwargs["drop_params"] is True
        assert kwargs["model"] == "voyage/voyage-3"
        n = len(kwargs["input"])
        return SimpleNamespace(
            model_dump=lambda: {"data": [{"embedding": [0.1, 0.2, 0.3]} for _ in range(n)]}
        )

    monkeypatch.setattr(
        "zotero_arxiv_daily.reranker.api.litellm", SimpleNamespace(embedding=fake_embedding)
    )
    reranker = ApiReranker(config)
    score = reranker.get_similarity_score(["hello", "world"], ["ping"])
    assert score.shape == (2, 1)
