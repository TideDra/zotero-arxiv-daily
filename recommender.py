import numpy as np
from sentence_transformers import SentenceTransformer
from paper import ArxivPaper, BiorxivPaper
from datetime import datetime
from loguru import logger

def rerank_paper(candidate:list[ArxivPaper],candidate_bio:list[BiorxivPaper],corpus:list[dict],model:str='avsolatorio/GIST-small-Embedding-v0') -> list[ArxivPaper]:
    if len(candidate) == 0 and len(candidate_bio) == 0:
        return candidate, candidate_bio
    encoder = SentenceTransformer(model)
    #sort corpus by date, from newest to oldest
    corpus = sorted(corpus,key=lambda x: datetime.strptime(x['data']['dateAdded'], '%Y-%m-%dT%H:%M:%SZ'),reverse=True)
    time_decay_weight = 1 / (1 + np.log10(np.arange(len(corpus)) + 1))
    time_decay_weight = time_decay_weight / time_decay_weight.sum()
    # calculate similarity according to the corpus' abstract and candidate's summary
    corpus_feature = encoder.encode([paper['data']['abstractNote'] for paper in corpus])
    
    if len(candidate) != 0:
        candidate_feature = encoder.encode([paper.summary for paper in candidate])
        sim = encoder.similarity(candidate_feature,corpus_feature) # [n_candidate, n_corpus]
        scores = (sim * time_decay_weight).sum(axis=1) * 10 # [n_candidate]
        for s,c in zip(scores,candidate):
            c.score = s.item()
        candidate = sorted(candidate,key=lambda x: x.score,reverse=True)
    if len(candidate_bio) != 0:
        candidate_bio_feature = encoder.encode([paper.summary for paper in candidate_bio])
        sim_bio = encoder.similarity(candidate_bio_feature,corpus_feature) # [n_candidate, n_corpus]
        scores_bio = (sim_bio * time_decay_weight).sum(axis=1) * 10 # [n_candidate]
        for s,c in zip(scores_bio,candidate_bio):
            c.score = s.item()
        candidate_bio = sorted(candidate_bio,key=lambda x: x.score,reverse=True)
    
    return candidate, candidate_bio
