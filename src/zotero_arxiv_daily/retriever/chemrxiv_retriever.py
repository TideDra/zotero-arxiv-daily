import html
import re
from datetime import datetime, timedelta, timezone
from time import sleep
from typing import Any

import requests
from loguru import logger

from .base import BaseRetriever, register_retriever
from ..protocol import Paper


@register_retriever("chemrxiv")
class ChemrxivRetriever(BaseRetriever):
    """Retrieve recent chemRxiv preprints via the Crossref REST API.

    chemRxiv moved from Cambridge Open Engage to Wiley's Research Exchange platform
    in January 2026; the old ``public-api`` is gone and the new site sits behind
    Cloudflare bot protection. Every chemRxiv preprint is registered with Crossref
    under the ACS prefix ``10.26434`` within minutes of posting, with title,
    abstract, authors, affiliations and links, so Crossref is used as the source.

    chemRxiv publishes continuously rather than in daily batches, so this retriever
    keeps every record whose Crossref ``created`` timestamp (the deposit time, which
    coincides with posting) falls within the last ``lookback_hours`` hours. Crossref
    does not expose chemRxiv subject categories, so there is no category filter;
    chemRxiv volume is small (a few dozen preprints per day) and the reranker does
    the selection.
    """

    api_url = "https://api.crossref.org/prefixes/10.26434/works"
    page_size = 100
    max_pages = 20
    lookback_hours = 24
    request_headers = {
        "User-Agent": "zotero-arxiv-daily (https://github.com/TideDra/zotero-arxiv-daily)",
        "Accept": "application/json",
    }
    _version_suffix = re.compile(r"[/-]v(\d+)$", re.IGNORECASE)
    _tag = re.compile(r"<[^>]+>")

    def _get_json(self, params: dict[str, Any]) -> dict[str, Any]:
        retry_num = 10
        delay_time = 10
        for i in range(retry_num):
            try:
                response = requests.get(
                    self.api_url,
                    params=params,
                    headers=self.request_headers,
                    timeout=60,
                )
                response.raise_for_status()
                return response.json()
            except Exception as exc:
                if i == retry_num - 1:
                    raise
                logger.warning(f"Failed to retrieve papers: {exc}. Retry in {delay_time} seconds.")
                sleep(delay_time)

    @staticmethod
    def _parse_date(value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed

    def _is_new_version(self, doi: str) -> bool:
        match = self._version_suffix.search(doi)
        return match is not None and int(match.group(1)) > 1

    def _retrieve_raw_papers(self) -> list[dict[str, Any]]:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=self.lookback_hours)
        include_new_versions = bool(getattr(self.retriever_config, "include_new_versions", False))

        collection = []
        for page in range(self.max_pages):
            result = self._get_json(
                {
                    "filter": f"type:posted-content,from-created-date:{cutoff.date().isoformat()}",
                    "sort": "created",
                    "order": "desc",
                    "rows": self.page_size,
                    "offset": page * self.page_size,
                }
            )
            items = result.get("message", {}).get("items", [])
            if not items:
                break
            reached_cutoff = False
            for item in items:
                created = self._parse_date(item.get("created", {}).get("date-time"))
                if created is None:
                    logger.warning(f"Skipping chemRxiv record without a valid created date: {item.get('DOI')}")
                    continue
                if created < cutoff:
                    reached_cutoff = True
                    break
                if not include_new_versions and self._is_new_version(item.get("DOI", "")):
                    continue
                collection.append(item)
            if reached_cutoff or len(items) < self.page_size:
                break

        if len(collection) == 0:
            logger.warning(f"No chemRxiv paper found in the last {self.lookback_hours} hours.")
        if self.config.executor.debug:
            collection = collection[:10]
        return collection

    @classmethod
    def _clean_text(cls, text: str | None) -> str:
        if not text:
            return ""
        text = cls._tag.sub(" ", text)
        text = html.unescape(text)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _author_name(author: dict[str, Any]) -> str:
        if author.get("name"):
            return author["name"].strip()
        return f"{author.get('given', '')} {author.get('family', '')}".strip()

    def convert_to_paper(self, raw_paper: dict[str, Any]) -> Paper | None:
        doi = raw_paper["DOI"]
        titles = raw_paper.get("title") or []
        title = self._clean_text(titles[0] if titles else "")
        if not title:
            return None
        authors = [self._author_name(a) for a in raw_paper.get("author", [])]
        authors = [a for a in authors if a]
        abstract = self._clean_text(raw_paper.get("abstract"))
        url = (raw_paper.get("resource") or {}).get("primary", {}).get("URL") or f"https://doi.org/{doi}"
        pdf_url = f"https://chemrxiv.org/doi/pdf/{doi}"
        full_text = None  # chemrxiv.org sits behind Cloudflare; do not scrape the PDF
        return Paper(
            source=self.name,
            title=title,
            authors=authors,
            abstract=abstract,
            url=url,
            pdf_url=pdf_url,
            full_text=full_text,
        )
