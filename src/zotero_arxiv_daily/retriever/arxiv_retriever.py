from .base import BaseRetriever, register_retriever
import arxiv
from arxiv import Result as ArxivResult
from ..protocol import Paper
from ..utils import extract_markdown_from_pdf, extract_tex_code_from_tar
from tempfile import TemporaryDirectory
from multiprocessing import get_context
from queue import Empty
from dataclasses import dataclass
import feedparser
from urllib.request import urlopen
from tqdm import tqdm
import os
import time
import shutil
from loguru import logger

PDF_EXTRACT_TIMEOUT = 180
DOWNLOAD_TIMEOUT = 30
INITIAL_BATCH_SIZE = 20
MAX_BATCH_FETCH_ATTEMPTS = 4
BACKOFF_BASE_SECONDS = 15
INTER_BATCH_DELAY_SECONDS = 3


@dataclass
class RSSPaper:
    entry_id: str
    title: str
    summary: str
    authors: list[str]
    pdf_url: str | None

    def source_url(self) -> str:
        return f"https://arxiv.org/e-print/{self.entry_id}"


@register_retriever("arxiv")
class ArxivRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        if self.config.source.arxiv.category is None:
            raise ValueError("category must be specified for arxiv.")

    def _retrieve_raw_papers(self) -> list[ArxivResult | RSSPaper]:
        client = arxiv.Client(num_retries=10, delay_seconds=10)
        query = '+'.join(self.config.source.arxiv.category)
        include_cross_list = self.config.source.arxiv.get("include_cross_list", False)

        # Get the latest paper IDs from the arXiv RSS feed.
        feed = feedparser.parse(f"https://rss.arxiv.org/atom/{query}")
        if 'Feed error for query' in feed.feed.title:
            raise Exception(f"Invalid ARXIV_QUERY: {query}.")
        allowed_announce_types = {"new", "cross"} if include_cross_list else {"new"}
        relevant_entries = [
            entry for entry in feed.entries
            if entry.get("arxiv_announce_type", "new") in allowed_announce_types
        ]
        all_paper_ids = [entry.id.removeprefix("oai:arXiv.org:") for entry in relevant_entries]

        if self.config.executor.debug:
            all_paper_ids = all_paper_ids[:10]
            relevant_entries = relevant_entries[:10]

        if len(all_paper_ids) == 0:
            logger.info(
                "arXiv RSS feed returned no matching entries. "
                "This usually means arXiv has not published a new batch yet, or today is a weekend/holiday."
            )
            return []

        rss_entries_by_id = {
            entry.id.removeprefix("oai:arXiv.org:"): entry for entry in relevant_entries
        }

        # Fetch full metadata from the API in batches, but degrade gracefully on rate limits.
        raw_papers = []
        bar = tqdm(total=len(all_paper_ids))
        for i in range(0, len(all_paper_ids), INITIAL_BATCH_SIZE):
            batch_ids = all_paper_ids[i:i + INITIAL_BATCH_SIZE]
            batch = self._fetch_batch_with_fallback(client, batch_ids, rss_entries_by_id)
            bar.update(len(batch))
            raw_papers.extend(batch)
            if i + INITIAL_BATCH_SIZE < len(all_paper_ids):
                time.sleep(INTER_BATCH_DELAY_SECONDS)
        bar.close()

        return raw_papers

    def _fetch_batch_with_fallback(
        self,
        client: arxiv.Client,
        batch_ids: list[str],
        rss_entries_by_id: dict[str, feedparser.FeedParserDict],
    ) -> list[ArxivResult | RSSPaper]:
        batch_size = len(batch_ids)
        current_ids = list(batch_ids)

        for attempt in range(MAX_BATCH_FETCH_ATTEMPTS):
            try:
                search = arxiv.Search(id_list=current_ids)
                return list(client.results(search))
            except arxiv.HTTPError as err:
                if not self._is_retryable_api_error(err):
                    raise

                if batch_size > 1:
                    batch_size = max(1, batch_size // 2)
                    logger.warning(
                        f"arXiv API returned HTTP {err.status} for a batch of {len(current_ids)} papers. "
                        f"Retrying with smaller batch size {batch_size}."
                    )
                    return self._fetch_split_batches(client, current_ids, rss_entries_by_id, batch_size)

                if attempt < MAX_BATCH_FETCH_ATTEMPTS - 1:
                    sleep_seconds = BACKOFF_BASE_SECONDS * (attempt + 1)
                    logger.warning(
                        f"arXiv API returned HTTP {err.status} for single-paper lookup {current_ids[0]}. "
                        f"Retrying in {sleep_seconds}s."
                    )
                    time.sleep(sleep_seconds)
                    continue

                logger.warning(
                    f"arXiv API kept returning HTTP {err.status} for {current_ids[0]}. "
                    "Falling back to RSS metadata for this paper."
                )
                return self._build_rss_fallback_batch(current_ids, rss_entries_by_id)

        return self._build_rss_fallback_batch(current_ids, rss_entries_by_id)

    def _is_retryable_api_error(self, err: arxiv.HTTPError) -> bool:
        return err.status == 429 or 500 <= err.status < 600

    def _fetch_split_batches(
        self,
        client: arxiv.Client,
        batch_ids: list[str],
        rss_entries_by_id: dict[str, feedparser.FeedParserDict],
        batch_size: int,
    ) -> list[ArxivResult | RSSPaper]:
        raw_papers = []
        for i in range(0, len(batch_ids), batch_size):
            sub_batch_ids = batch_ids[i:i + batch_size]
            raw_papers.extend(self._fetch_batch_with_fallback(client, sub_batch_ids, rss_entries_by_id))
            if i + batch_size < len(batch_ids):
                time.sleep(INTER_BATCH_DELAY_SECONDS)
        return raw_papers

    def _build_rss_fallback_batch(
        self,
        batch_ids: list[str],
        rss_entries_by_id: dict[str, feedparser.FeedParserDict],
    ) -> list[RSSPaper]:
        fallback_papers = []
        for paper_id in batch_ids:
            entry = rss_entries_by_id.get(paper_id)
            if entry is None:
                logger.warning(f"Paper {paper_id} was missing from RSS fallback cache and will be skipped.")
                continue
            fallback_papers.append(self._build_rss_paper(entry))
        return fallback_papers

    def _build_rss_paper(self, entry: feedparser.FeedParserDict) -> RSSPaper:
        paper_id = entry.id.removeprefix("oai:arXiv.org:")
        abstract = entry.summary
        if "Abstract:" in abstract:
            abstract = abstract.split("Abstract:", 1)[1].strip()

        authors = []
        if creators := entry.get("dc_creator"):
            authors = [author.strip() for author in creators.split(",") if author.strip()]
        elif author_detail := entry.get("author_detail"):
            if name := author_detail.get("name"):
                authors = [name]

        return RSSPaper(
            entry_id=paper_id,
            title=entry.title,
            summary=abstract,
            authors=authors,
            pdf_url=f"https://arxiv.org/pdf/{paper_id}.pdf",
        )

    def convert_to_paper(self, raw_paper: ArxivResult | RSSPaper) -> Paper:
        title = raw_paper.title
        authors = raw_paper.authors if isinstance(raw_paper, RSSPaper) else [a.name for a in raw_paper.authors]
        abstract = raw_paper.summary
        pdf_url = raw_paper.pdf_url
        full_text = extract_text_from_pdf_with_timeout(raw_paper)
        if full_text is None:
            full_text = extract_text_from_tar(raw_paper)
        return Paper(
            source=self.name,
            title=title,
            authors=authors,
            abstract=abstract,
            url=raw_paper.entry_id,
            pdf_url=pdf_url,
            full_text=full_text
        )

def extract_text_from_pdf_with_timeout(paper: ArxivResult | RSSPaper) -> str | None:
    ctx = get_context("fork" if os.name == "posix" else "spawn")
    queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(target=_extract_text_from_pdf_worker, args=(paper, queue))
    proc.start()
    proc.join(PDF_EXTRACT_TIMEOUT)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        logger.warning(f"PDF extraction timed out for {paper.title}")
        return None

    try:
        ok, payload = queue.get_nowait()
    except Empty:
        logger.warning(f"PDF extraction exited without returning content for {paper.title}")
        return None

    if ok:
        return payload

    logger.warning(f"Failed to extract full text of {paper.title} from pdf: {payload}")
    return None

def extract_text_from_pdf(paper: ArxivResult) -> str | None:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.pdf")
        if paper.pdf_url is None:
            logger.warning(f"No PDF URL available for {paper.title}")
            return None
        download_url_to_file(paper.pdf_url, path, DOWNLOAD_TIMEOUT)
        return extract_markdown_from_pdf(path)

def _extract_text_from_pdf_worker(paper: ArxivResult | RSSPaper, queue) -> None:
    try:
        queue.put((True, extract_text_from_pdf(paper)))
    except Exception as e:
        queue.put((False, str(e)))

def extract_text_from_tar(paper: ArxivResult) -> str | None:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.tar.gz")
        source_url = paper.source_url()
        if source_url is None:
            logger.warning(f"No source URL available for {paper.title}")
            return None
        download_url_to_file(source_url, path, DOWNLOAD_TIMEOUT)
        try:
            file_contents = extract_tex_code_from_tar(path, paper.entry_id)
            if "all" not in file_contents:
                logger.warning(f"Failed to extract full text of {paper.title} from tar: Main tex file not found.")
                return None
            full_text = file_contents["all"]
        except Exception as e:
            logger.warning(f"Failed to extract full text of {paper.title} from tar: {e}")
            full_text = None
        return full_text

def download_url_to_file(url: str, path: str, timeout: int) -> None:
    with urlopen(url, timeout=timeout) as response, open(path, "wb") as out_file:
        shutil.copyfileobj(response, out_file)
