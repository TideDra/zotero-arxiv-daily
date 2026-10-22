from .biorxiv_retriever import BiorxivRetriever
from .base import register_retriever

@register_retriever("medrxiv")
class MedrxivRetriever(BiorxivRetriever):
    server = "medrxiv"