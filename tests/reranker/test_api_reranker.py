from zotero_arxiv_daily.reranker.api import ApiReranker

def test_api_reranker(config):
    reranker = ApiReranker(config)
    score = reranker.get_similarity_score(["hello","world"], ["ping"])
    assert score.shape == (2,1)