from .base import BaseRetriever, register_retriever
import arxiv
from arxiv import Result as ArxivResult
from ..protocol import Paper
from ..utils import extract_markdown_from_pdf, extract_tex_code_from_tar
from tempfile import TemporaryDirectory
import feedparser
import multiprocessing
import os
from queue import Empty
from time import sleep
from typing import Any, Callable, TypeVar
from loguru import logger
import requests

T = TypeVar("T")

DOWNLOAD_TIMEOUT = (10, 60)
PDF_EXTRACT_TIMEOUT = 180
TAR_EXTRACT_TIMEOUT = 180


def canonical_arxiv_id(identifier: str) -> str:
    value = identifier.removeprefix("arXiv:")
    value = value.split("/abs/")[-1]
    return value.rsplit("v", 1)[0] if value.rsplit("v", 1)[-1].isdigit() else value


def fetch_arxiv_results_by_ids(
    paper_ids: list[str],
    client: arxiv.Client | None = None,
) -> list[ArxivResult]:
    """Fetch arXiv metadata in bounded batches for retrieval or SciX enrichment."""
    if not paper_ids:
        return []
    client = client or arxiv.Client(num_retries=10, delay_seconds=10)
    raw_papers: list[ArxivResult] = []
    max_batch_retries = 5
    batch_retry_delay = 30
    for i in range(0, len(paper_ids), 20):
        search = arxiv.Search(id_list=paper_ids[i:i + 20])
        for attempt in range(max_batch_retries):
            try:
                raw_papers.extend(client.results(search))
                break
            except arxiv.HTTPError as exc:
                if exc.status == 429 and attempt < max_batch_retries - 1:
                    wait = batch_retry_delay * (attempt + 1)
                    logger.warning(
                        f"arXiv API 429 on batch {i // 20}, "
                        f"retry {attempt + 1}/{max_batch_retries} in {wait}s"
                    )
                    sleep(wait)
                else:
                    raise
        if i + 20 < len(paper_ids):
            sleep(3)
    return raw_papers


def _download_file(url: str, path: str) -> None:
    with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT) as response:
        response.raise_for_status()
        with open(path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)


def _run_in_subprocess(
    result_queue: Any,
    func: Callable[..., T | None],
    args: tuple[Any, ...],
) -> None:
    try:
        result_queue.put(("ok", func(*args)))
    except Exception as exc:
        result_queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _run_with_hard_timeout(
    func: Callable[..., T | None],
    args: tuple[Any, ...],
    *,
    timeout: float,
    operation: str,
    paper_title: str,
) -> T | None:
    start_methods = multiprocessing.get_all_start_methods()
    context = multiprocessing.get_context("fork" if "fork" in start_methods else start_methods[0])
    result_queue = context.Queue()
    process = context.Process(target=_run_in_subprocess, args=(result_queue, func, args))
    process.start()

    try:
        status, payload = result_queue.get(timeout=timeout)
    except Empty:
        if process.is_alive():
            process.kill()
        process.join(5)
        result_queue.close()
        result_queue.join_thread()
        logger.warning(f"{operation} timed out for {paper_title} after {timeout} seconds")
        return None

    process.join(5)
    result_queue.close()
    result_queue.join_thread()

    if status == "ok":
        return payload

    logger.warning(f"{operation} failed for {paper_title}: {payload}")
    return None


def _extract_text_from_pdf_worker(pdf_url: str) -> str:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.pdf")
        _download_file(pdf_url, path)
        return extract_markdown_from_pdf(path)


def _extract_text_from_html_worker(html_url: str) -> str | None:
    import trafilatura

    downloaded = trafilatura.fetch_url(html_url)
    if downloaded is None:
        raise ValueError(f"Failed to download HTML from {html_url}")
    text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
    if not text:
        raise ValueError(f"No text extracted from {html_url}")
    return text


def _extract_text_from_tar_worker(source_url: str, paper_id: str, paper_title: str | None = None) -> str | None:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.tar.gz")
        _download_file(source_url, path)
        file_contents = extract_tex_code_from_tar(path, paper_id, paper_title=paper_title)
        if not file_contents or "all" not in file_contents:
            raise ValueError("Main tex file not found.")
        return file_contents["all"]


@register_retriever("arxiv")
class ArxivRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        if self.config.source.arxiv.category is None:
            raise ValueError("category must be specified for arxiv.")

    def _retrieve_raw_papers(self) -> list[ArxivResult]:
        query = '+'.join(self.config.source.arxiv.category)
        include_cross_list = self.config.source.arxiv.get("include_cross_list", False)
        # Get the latest paper from arxiv rss feed
        feed = feedparser.parse(f"https://rss.arxiv.org/atom/{query}")
        if 'Feed error for query' in feed.feed.title:
            raise Exception(f"Invalid ARXIV_QUERY: {query}.")
        raw_papers = []
        allowed_announce_types = {"new", "cross"} if include_cross_list else {"new"}
        all_paper_ids = [
            i.id.removeprefix("oai:arXiv.org:")
            for i in feed.entries
            if i.get("arxiv_announce_type", "new") in allowed_announce_types
        ]
        if self.config.executor.debug:
            all_paper_ids = all_paper_ids[:10]

        # Get full information of each paper from arxiv api
        raw_papers = fetch_arxiv_results_by_ids(all_paper_ids)
        return raw_papers

    def convert_to_paper(self, raw_paper: ArxivResult) -> Paper:
        return convert_arxiv_result_to_paper(raw_paper, source=self.name)


def extract_text_from_html(paper: ArxivResult) -> str | None:
    html_url = paper.entry_id.replace("/abs/", "/html/")
    try:
        return _extract_text_from_html_worker(html_url)
    except Exception as exc:
        logger.warning(f"HTML extraction failed for {paper.title}: {exc}")
        return None


def extract_text_from_pdf(paper: ArxivResult) -> str | None:
    if paper.pdf_url is None:
        logger.warning(f"No PDF URL available for {paper.title}")
        return None
    return _run_with_hard_timeout(
        _extract_text_from_pdf_worker,
        (paper.pdf_url,),
        timeout=PDF_EXTRACT_TIMEOUT,
        operation="PDF extraction",
        paper_title=paper.title,
    )


def extract_text_from_tar(paper: ArxivResult) -> str | None:
    source_url = paper.source_url()
    if source_url is None:
        logger.warning(f"No source URL available for {paper.title}")
        return None
    return _run_with_hard_timeout(
        _extract_text_from_tar_worker,
        (source_url, paper.entry_id, paper.title),
        timeout=TAR_EXTRACT_TIMEOUT,
        operation="Tar extraction",
        paper_title=paper.title,
    )


def convert_arxiv_result_to_paper(raw_paper: ArxivResult, source: str = "arxiv") -> Paper:
    full_text = extract_text_from_tar(raw_paper)
    if full_text is None:
        full_text = extract_text_from_html(raw_paper)
    if full_text is None:
        full_text = extract_text_from_pdf(raw_paper)
    short_id = raw_paper.get_short_id() if hasattr(raw_paper, "get_short_id") else raw_paper.entry_id
    arxiv_id = canonical_arxiv_id(short_id)
    external_ids = {"arxiv": arxiv_id}
    if getattr(raw_paper, "doi", None):
        external_ids["doi"] = raw_paper.doi
    return Paper(
        source=source,
        title=raw_paper.title,
        authors=[author.name for author in raw_paper.authors],
        abstract=raw_paper.summary,
        url=raw_paper.entry_id,
        external_ids=external_ids,
        published_at=getattr(raw_paper, "published", None),
        keywords=list(getattr(raw_paper, "categories", None) or []),
        pdf_url=raw_paper.pdf_url,
        full_text=full_text,
        content_source="arxiv",
        remote_processing_allowed=True,
    )


def enrich_papers_from_arxiv(papers: list[Paper]) -> None:
    """Replace SciX/ADS text with arXiv-sourced content for selected mapped records."""
    mapped = [paper for paper in papers if paper.external_ids.get("arxiv")]
    if not mapped:
        return
    results = fetch_arxiv_results_by_ids([paper.external_ids["arxiv"] for paper in mapped])
    by_id = {canonical_arxiv_id(result.get_short_id()): result for result in results}
    for paper in mapped:
        arxiv_id = paper.external_ids["arxiv"]
        raw_paper = by_id.get(canonical_arxiv_id(arxiv_id))
        if raw_paper is None:
            logger.warning(f"Could not enrich SciX record {paper.stable_id} from arXiv {arxiv_id}")
            continue
        enriched = convert_arxiv_result_to_paper(raw_paper)
        paper.title = enriched.title
        paper.authors = enriched.authors
        paper.abstract = enriched.abstract
        paper.full_text = enriched.full_text
        paper.pdf_url = enriched.pdf_url
        paper.published_at = paper.published_at or enriched.published_at
        paper.external_ids.update(enriched.external_ids)
        paper.content_source = "arxiv"
        paper.remote_processing_allowed = True
