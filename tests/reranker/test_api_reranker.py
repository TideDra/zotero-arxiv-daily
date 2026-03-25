import pytest
from zotero_arxiv_daily.reranker.api import ApiReranker

@pytest.mark.ci
def test_api_reranker(config):
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
