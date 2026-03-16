from zotero_arxiv_daily.reranker.local import LocalReranker

def test_local_reranker(config):
    reranker = LocalReranker(config)
    score = reranker.get_similarity_score(["hello","world"], ["ping"])
    assert score.shape == (2,1)