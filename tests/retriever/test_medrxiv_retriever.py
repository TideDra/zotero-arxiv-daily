"""Tests for MedrxivRetriever."""

from omegaconf import open_dict

from zotero_arxiv_daily.retriever.medrxiv_retriever import MedrxivRetriever


def test_medrxiv_server_attribute(config):
    with open_dict(config.source):
        config.source.medrxiv = {"category": ["neurology"]}
    retriever = MedrxivRetriever(config)
    assert retriever.server == "medrxiv"


def test_medrxiv_pdf_url(config):
    with open_dict(config.source):
        config.source.medrxiv = {"category": ["neurology"]}
    retriever = MedrxivRetriever(config)
    paper = retriever.convert_to_paper({
        "doi": "10.1101/2026.03.01.999",
        "title": "A medrxiv paper",
        "authors": "Smith, J.",
        "abstract": "Abstract.",
        "version": "1",
    })
    assert "medrxiv.org" in paper.pdf_url
    assert paper.source == "medrxiv"
