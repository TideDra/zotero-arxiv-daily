from .base import BaseRetriever, register_retriever
import arxiv
from arxiv import Result as ArxivResult
from ..protocol import Paper
from ..utils import extract_markdown_from_pdf, extract_tex_code_from_tar
from datetime import datetime, timedelta, timezone
import multiprocessing
import os
from queue import Empty
import re
from tempfile import TemporaryDirectory
from time import sleep
from typing import Any, Callable, TypeVar
import urllib.request
import xml.etree.ElementTree as ET
from loguru import logger
import requests
import feedparser
from tqdm import tqdm

T = TypeVar("T")

DOWNLOAD_TIMEOUT = (10, 60)
PDF_EXTRACT_TIMEOUT = 180
TAR_EXTRACT_TIMEOUT = 180


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


def _clean_abstract(summary_raw: str) -> str:
    cleaned = re.sub(
        r"^arxiv:[^\n]+\n(?:announce type:[^\n]+\n)?abstract:\s*",
        "",
        summary_raw.strip(),
        flags=re.IGNORECASE,
    ).strip()
    if "Abstract:" in cleaned:
        cleaned = cleaned.split("Abstract:", 1)[1].strip()
    return " ".join(cleaned.split())


def _extract_authors(entry: Any) -> list[ArxivResult.Author]:
    author_str = getattr(entry, "author", "") or ""
    if not author_str and hasattr(entry, "authors") and entry.authors:
        author_str = ", ".join(
            a.get("name", "") if isinstance(a, dict) else getattr(a, "name", str(a))
            for a in entry.authors
        )
    author_names = [a.strip() for a in author_str.split(",") if a.strip()]
    if not author_names:
        author_names = ["Unknown"]
    return [ArxivResult.Author(name=name) for name in author_names]


def _entry_to_arxiv_result(entry: Any) -> ArxivResult:
    raw_id = getattr(entry, "id", "") or ""
    paper_id = raw_id.removeprefix("oai:arXiv.org:")
    title_raw = getattr(entry, "title", "") or ""
    title = " ".join(title_raw.split())

    authors = _extract_authors(entry)
    summary = _clean_abstract(getattr(entry, "summary", "") or "")

    link = getattr(entry, "link", "") or f"https://arxiv.org/abs/{paper_id}"
    pdf_url = f"https://arxiv.org/pdf/{paper_id}"

    result = ArxivResult(
        entry_id=link,
        title=title,
        authors=authors,
        summary=summary,
        links=[ArxivResult.Link(href=pdf_url, title="pdf")],
    )
    result.pdf_url = pdf_url
    return result


def _fetch_oaipmh_papers(categories: list[str], days_back: int) -> list[ArxivResult]:
    from_date = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%d")
    target_cats = set(categories)
    sets = sorted(list({c.split(".")[0] for c in categories}))
    results = []
    seen = set()

    for s in sets:
        url = f"https://oaipmh.arxiv.org/oai?verb=ListRecords&metadataPrefix=arXivRaw&set={s}&from={from_date}"
        logger.info(f"Fetching OAI-PMH records from: {url} (from {from_date})")
        retry_num = 3
        delay_time = 5
        records = []
        for attempt in range(retry_num):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "zotero-arxiv-daily"})
                with urllib.request.urlopen(req, timeout=60) as resp:
                    root = ET.fromstring(resp.read())
                records = root.findall(".//{*}record")
                break
            except Exception as exc:
                if attempt == retry_num - 1:
                    logger.warning(f"Failed to fetch OAI-PMH records for set {s}: {exc}")
                else:
                    sleep(delay_time)

        for r in records:
            header = r.find(".//{*}header")
            if header is not None and header.get("status") == "deleted":
                continue
            meta = r.find(".//{*}arXivRaw")
            if meta is None:
                continue
            paper_cats = (meta.findtext("{*}categories") or "").split()
            if not any(c in target_cats for c in paper_cats):
                continue
            id_el = header.find("{*}identifier") if header is not None else None
            if id_el is None or not id_el.text:
                continue
            raw_id = id_el.text.removeprefix("oai:arXiv.org:")
            paper_key = raw_id.split("v")[0]
            if paper_key in seen:
                continue
            seen.add(paper_key)

            title = " ".join((meta.findtext("{*}title") or "").split())
            author_str = meta.findtext("{*}authors") or ""
            authors = [ArxivResult.Author(name=a.strip()) for a in author_str.split(",") if a.strip()]
            if not authors:
                authors = [ArxivResult.Author(name="Unknown")]
            summary = " ".join((meta.findtext("{*}abstract") or "").split())

            link = f"https://arxiv.org/abs/{raw_id}"
            pdf_url = f"https://arxiv.org/pdf/{raw_id}"

            res = ArxivResult(
                entry_id=link,
                title=title,
                authors=authors,
                summary=summary,
                links=[ArxivResult.Link(href=pdf_url, title="pdf")],
            )
            res.pdf_url = pdf_url
            results.append(res)

    return results


@register_retriever("arxiv")
class ArxivRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        if self.config.source.arxiv.category is None:
            raise ValueError("category must be specified for arxiv.")

    def _retrieve_raw_papers(self) -> list[ArxivResult]:
        query = '+'.join(self.config.source.arxiv.category)
        include_cross_list = self.config.source.arxiv.get("include_cross_list", False)
        days_back = int(self.config.source.arxiv.get("days_back", 1) or 1)
        rss_url = f"https://rss.arxiv.org/atom/{query}"

        # Get latest papers from arxiv RSS feed with retry
        retry_num = 5
        delay_time = 5
        feed = None
        for attempt in range(retry_num):
            feed = feedparser.parse(rss_url)
            if hasattr(feed, "feed") and hasattr(feed.feed, "title"):
                if "Feed error for query" in feed.feed.title:
                    raise Exception(f"Invalid ARXIV_QUERY: {query}.")
                break
            if attempt < retry_num - 1:
                logger.warning(f"Failed to fetch arxiv RSS feed, retrying in {delay_time}s...")
                sleep(delay_time)
        else:
            if feed is None or not getattr(feed, "entries", None):
                raise RuntimeError(f"Failed to fetch arxiv RSS feed from {rss_url}")

        allowed_announce_types = {"new", "cross"} if include_cross_list else {"new"}
        target_entries = [
            i for i in feed.entries
            if i.get("arxiv_announce_type", "new") in allowed_announce_types
        ]

        seen_paper_ids = set()
        raw_papers = []
        for entry in target_entries:
            raw_id = getattr(entry, "id", "") or ""
            paper_id = raw_id.removeprefix("oai:arXiv.org:")
            paper_key = paper_id.split("v")[0]
            if paper_key and paper_key in seen_paper_ids:
                continue
            if paper_key:
                seen_paper_ids.add(paper_key)
            raw_papers.append(_entry_to_arxiv_result(entry))

        if days_back > 1:
            logger.info(f"Retrieving older papers via OAI-PMH (past {days_back} days)...")
            oaipmh_papers = _fetch_oaipmh_papers(self.config.source.arxiv.category, days_back)
            added_count = 0
            for p in oaipmh_papers:
                paper_key = p.get_short_id().split("v")[0]
                if paper_key and paper_key not in seen_paper_ids:
                    seen_paper_ids.add(paper_key)
                    raw_papers.append(p)
                    added_count += 1
            logger.info(f"Added {added_count} older papers from OAI-PMH. Total raw papers: {len(raw_papers)}")

        if self.config.executor.debug:
            max_debug = int(os.environ.get("MAX_DEBUG_PAPERS", 10))
            raw_papers = raw_papers[:max_debug]

        return raw_papers

    def convert_to_paper(self, raw_paper: ArxivResult) -> Paper:
        title = raw_paper.title
        authors = [a.name for a in raw_paper.authors]
        abstract = raw_paper.summary
        pdf_url = raw_paper.pdf_url
        full_text = extract_text_from_tar(raw_paper)
        if full_text is None:
            full_text = extract_text_from_html(raw_paper)
        if full_text is None:
            full_text = extract_text_from_pdf(raw_paper)
        return Paper(
            source=self.name,
            title=title,
            authors=authors,
            abstract=abstract,
            url=raw_paper.entry_id,
            pdf_url=pdf_url,
            full_text=full_text,
        )


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
