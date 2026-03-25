from types import SimpleNamespace

import numpy as np

import zotero_arxiv_daily.reranker.api as api_module
from zotero_arxiv_daily.reranker.api import ApiReranker


def test_api_reranker(config, monkeypatch):
    observed = {}

    class FakeOpenAI:
        def __init__(self, *args, **kwargs):
            observed["init_kwargs"] = kwargs
            self.embeddings = SimpleNamespace(create=self.create)

        def create(self, *, input, model):
            observed["request"] = {"input": input, "model": model}
            embedding_map = {
                "hello": [1.0, 0.0],
                "world": [0.0, 1.0],
                "ping": [1.0, 1.0],
            }
            return SimpleNamespace(
                data=[SimpleNamespace(embedding=embedding_map[text]) for text in input]
            )

    monkeypatch.setattr(api_module, "OpenAI", FakeOpenAI)

    reranker = ApiReranker(config)
    score = reranker.get_similarity_score(["hello", "world"], ["ping"])

    assert score.shape == (2, 1)
    assert np.allclose(score, np.array([[1 / np.sqrt(2)], [1 / np.sqrt(2)]]))
    assert observed["init_kwargs"] == {
        "api_key": config.reranker.api.key,
        "base_url": config.reranker.api.base_url,
    }
    assert observed["request"] == {
        "input": ["hello", "world", "ping"],
        "model": config.reranker.api.model,
    }
