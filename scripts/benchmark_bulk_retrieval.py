"""Benchmark and validation script for bulk arXiv retrieval.

Tests retrieval and parsing of large numbers of arXiv papers
via either multi-category RSS or OAI-PMH (for multi-day retrieval),
validating that metadata parsing is robust and free from HTTP 406 errors.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import re
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET

import feedparser
from loguru import logger

from zotero_arxiv_daily.retriever.arxiv_retriever import (
    _entry_to_arxiv_result,
    extract_text_from_html,
    extract_text_from_pdf,
    extract_text_from_tar,
)


def benchmark_rss_bulk(categories: list[str], include_replace: bool = True) -> list[any]:
    query = "+".join(categories)
    url = f"https://rss.arxiv.org/atom/{query}"
    logger.info(f"Fetching RSS feed from: {url}")
    t0 = time.time()
    feed = feedparser.parse(url)
    elapsed = time.time() - t0

    if not hasattr(feed, "entries") or not feed.entries:
        raise RuntimeError(f"Failed to fetch RSS entries from {url}")

    logger.info(f"Fetched {len(feed.entries)} total entries in {elapsed:.2f}s")

    allowed_types = {"new", "cross"}
    if include_replace:
        allowed_types.update({"replace", "replace-cross"})

    filtered = [
        e for e in feed.entries
        if e.get("arxiv_announce_type", "new") in allowed_types
    ]
    logger.info(f"Filtered entries matching {allowed_types}: {len(filtered)}")

    results = []
    seen = set()
    for entry in filtered:
        pid = getattr(entry, "id", "").removeprefix("oai:arXiv.org:")
        if pid in seen:
            continue
        seen.add(pid)
        results.append(_entry_to_arxiv_result(entry))

    logger.info(f"Successfully parsed {len(results)} unique ArxivResult objects")
    return results


def benchmark_oaipmh_bulk(days_back: int = 2) -> list[any]:
    target_date = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%d")
    url = f"https://oaipmh.arxiv.org/oai?verb=ListRecords&metadataPrefix=arXivRaw&set=cs&from={target_date}"
    logger.info(f"Fetching OAI-PMH records from: {url} (from {target_date})")

    req = urllib.request.Request(url, headers={"User-Agent": "zotero-arxiv-daily-bulk-test"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=60) as resp:
        content = resp.read()
    elapsed = time.time() - t0

    root = ET.fromstring(content)
    records = root.findall(".//{*}record")
    logger.info(f"Fetched {len(records)} records via OAI-PMH in {elapsed:.2f}s")

    from arxiv import Result as ArxivResult

    results = []
    seen = set()
    for r in records:
        header = r.find(".//{*}header")
        if header is not None and header.get("status") == "deleted":
            continue
        id_el = header.find("{*}identifier") if header is not None else None
        if id_el is None or not id_el.text:
            continue
        raw_id = id_el.text.removeprefix("oai:arXiv.org:")
        paper_id = raw_id.split("v")[0]
        if paper_id in seen:
            continue
        seen.add(paper_id)

        meta = r.find(".//{*}arXivRaw")
        if meta is None:
            continue

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

    logger.info(f"Successfully parsed {len(results)} unique ArxivResult objects from past {days_back} days")
    return results


def verify_sample_conversion(papers: list[any], sample_size: int = 3) -> None:
    sample = papers[:sample_size]
    logger.info(f"Verifying full-text retrieval on sample of {len(sample)} papers...")
    for idx, paper in enumerate(sample, 1):
        logger.info(f"[{idx}/{len(sample)}] Testing {paper.get_short_id()}: {paper.title[:50]}...")
        text = extract_text_from_html(paper)
        if text is None:
            text = extract_text_from_pdf(paper)
        status = "SUCCESS" if text else "NONE (expected for some formats/preprints)"
        length = len(text) if text else 0
        logger.info(f"    Full-text status: {status}, characters: {length}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test bulk arXiv retrieval")
    parser.add_argument(
        "--mode",
        choices=["rss", "oaipmh", "all"],
        default="all",
        help="Retrieval mode to test",
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=2,
        help="Days back for OAI-PMH retrieval (default: 2)",
    )
    parser.add_argument(
        "--sample-conversion",
        type=int,
        default=3,
        help="Number of papers to sample for full-text extraction test",
    )
    args = parser.parse_args()

    total_tested = 0

    if args.mode in ("rss", "all"):
        logger.info("=== 1. Testing Multi-Category Large RSS Retrieval ===")
        # Subscribe to a broad set of CS & ML categories
        broad_cats = [
            "cs.AI", "cs.CV", "cs.LG", "cs.CL", "cs.RO",
            "cs.CR", "cs.SE", "cs.NE", "stat.ML", "math.PR"
        ]
        rss_papers = benchmark_rss_bulk(broad_cats, include_replace=True)
        total_tested += len(rss_papers)
        logger.success(f"RSS Bulk Test Passed: retrieved {len(rss_papers)} papers without HTTP 406.")

    if args.mode in ("oaipmh", "all"):
        logger.info(f"=== 2. Testing OAI-PMH Multi-Day Retrieval ({args.days_back} days) ===")
        oai_papers = benchmark_oaipmh_bulk(days_back=args.days_back)
        total_tested += len(oai_papers)
        logger.success(f"OAI-PMH Test Passed: retrieved {len(oai_papers)} papers without HTTP 406.")

    papers_to_sample = rss_papers if args.mode in ("rss", "all") else oai_papers
    if args.sample_conversion > 0 and papers_to_sample:
        verify_sample_conversion(papers_to_sample, args.sample_conversion)

    logger.success(f"ALL TESTS COMPLETED: Total {total_tested} bulk papers retrieved successfully!")


if __name__ == "__main__":
    main()
