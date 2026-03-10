from .base import BaseRetriever, register_retriever
import arxiv
from arxiv import Result as ArxivResult
from ..protocol import Paper
from ..utils import extract_markdown_from_pdf, extract_tex_code_from_tar
from tempfile import TemporaryDirectory
import feedparser
from urllib.request import urlretrieve
from tqdm import tqdm
import os
from loguru import logger
@register_retriever("arxiv")
class ArxivRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        if self.config.source.arxiv.category is None:
            raise ValueError("category must be specified for arxiv.")
    def _retrieve_raw_papers(self) -> list[ArxivResult]:
        client = arxiv.Client(num_retries=10,delay_seconds=10)
        query = '+'.join(self.config.source.arxiv.category)
        # Get the latest paper from arxiv rss feed
        feed = feedparser.parse(f"https://rss.arxiv.org/atom/{query}")
        if 'Feed error for query' in feed.feed.title:
            raise Exception(f"Invalid ARXIV_QUERY: {query}.")
        raw_papers = []
        all_paper_ids = [i.id.removeprefix("oai:arXiv.org:") for i in feed.entries if i.get("arxiv_announce_type","new") == 'new']
        if self.config.executor.debug:
            all_paper_ids = all_paper_ids[:10]

        # Get full information of each paper from arxiv api
        bar = tqdm(total=len(all_paper_ids))
        for i in range(0,len(all_paper_ids),20):
            search = arxiv.Search(id_list=all_paper_ids[i:i+20])
            batch = list(client.results(search))
            bar.update(len(batch))
            raw_papers.extend(batch)
        bar.close()

        return raw_papers

    def convert_to_paper(self, raw_paper:ArxivResult) -> Paper:
        return Paper(
            source=self.name,
            title=raw_paper.title,
            authors=[a.name for a in raw_paper.authors],
            abstract=raw_paper.summary,
            url=raw_paper.entry_id,
            pdf_url=raw_paper.pdf_url,
            full_text=None  # deferred: fetched after reranking via fetch_full_text()
        )

    def fetch_full_text(self, paper: Paper) -> None:
        source_url = paper.url.replace("http://arxiv.org/abs/", "https://arxiv.org/e-print/")
        full_text = extract_text_from_pdf(paper.pdf_url, paper.title)
        if full_text is None:
            full_text = extract_text_from_tar(source_url, paper.url, paper.title)
        paper.full_text = full_text


def extract_text_from_pdf(pdf_url: str | None, title: str = "") -> str | None:
    if pdf_url is None:
        logger.warning(f"No PDF URL available for {title}")
        return None
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.pdf")
        urlretrieve(pdf_url, path)
        try:
            full_text = extract_markdown_from_pdf(path)
        except Exception as e:
            logger.warning(f"Failed to extract full text of {title} from pdf: {e}")
            full_text = None
        return full_text


def extract_text_from_tar(source_url: str, entry_id: str, title: str = "") -> str | None:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.tar.gz")
        urlretrieve(source_url, path)
        try:
            file_contents = extract_tex_code_from_tar(path, entry_id)
            if "all" not in file_contents:
                logger.warning(f"Failed to extract full text of {title} from tar: Main tex file not found.")
                return None
            full_text = file_contents["all"]
        except Exception as e:
            logger.warning(f"Failed to extract full text of {title} from tar: {e}")
            full_text = None
        return full_text