from zotero_arxiv_daily.retriever.arxiv_retriever import ArxivRetriever
from zotero_arxiv_daily.retriever.base import BaseRetriever, register_retriever
from zotero_arxiv_daily.protocol import Paper
import feedparser
import io
from omegaconf import open_dict
from urllib.error import HTTPError

def test_arxiv_retriever(config, monkeypatch):

    parsed_result = feedparser.parse("tests/retriever/arxiv_rss_example.xml")
    raw_parser = feedparser.parse
    def mock_feedparser_parse(url):
        if url == f"https://rss.arxiv.org/atom/{'+'.join(config.source.arxiv.category)}":
            return parsed_result
        return raw_parser(url)
    monkeypatch.setattr(feedparser, "parse", mock_feedparser_parse)
    
    retriever = ArxivRetriever(config)
    papers = retriever.retrieve_papers()
    parsed_results = [i for i in parsed_result.entries if i.get("arxiv_announce_type","new") == 'new']
    assert len(papers) == len(parsed_results)
    paper_titles = [i.title for i in papers]
    parsed_titles = [i.title for i in parsed_results]
    assert set(paper_titles) == set(parsed_titles)


@register_retriever("failing_test")
class FailingTestRetriever(BaseRetriever):
    def _retrieve_raw_papers(self) -> list[dict[str, str]]:
        return [
            {"title": "good paper", "mode": "ok"},
            {"title": "bad paper", "mode": "fail"},
        ]

    def convert_to_paper(self, raw_paper: dict[str, str]) -> Paper | None:
        if raw_paper["mode"] == "fail":
            raise HTTPError(
                url="https://example.com/paper.pdf",
                code=404,
                msg="not found",
                hdrs=None,
                fp=io.BufferedReader(io.BytesIO(b"missing")),
            )
        return Paper(
            source=self.name,
            title=raw_paper["title"],
            authors=[],
            abstract="",
            url=f"https://example.com/{raw_paper['mode']}",
        )


def test_retrieve_papers_skips_non_pickleable_worker_errors(config):
    with open_dict(config.source):
        config.source.failing_test = {}
    config.executor.max_workers = 2

    retriever = FailingTestRetriever(config)

    papers = retriever.retrieve_papers()

    assert [paper.title for paper in papers] == ["good paper"]
