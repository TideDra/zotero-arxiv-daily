from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from loguru import logger

from ..protocol import Paper
from .base import BaseRetriever, register_retriever

OPENREVIEW_API2_NOTES_URL = "https://api2.openreview.net/notes"
OPENREVIEW_API1_NOTES_URL = "https://api.openreview.net/notes"
OPENREVIEW_FORUM_URL = "https://openreview.net/forum?id={id}"
OPENREVIEW_PDF_URL = "https://openreview.net/pdf?id={id}"


@dataclass
class OpenReviewPaperItem:
    note: dict[str, Any]
    venue: str


@register_retriever("conference")
class ConferenceRetriever(BaseRetriever):
    """Retrieve accepted conference papers from OpenReview venues."""

    def __init__(self, config):
        super().__init__(config)
        if not self.retriever_config.venues:
            raise ValueError("venues must be specified for conference.")

    def _retrieve_raw_papers(self) -> list[OpenReviewPaperItem]:
        raw_papers: list[OpenReviewPaperItem] = []
        seen_note_ids: set[str] = set()
        limit = int(self.retriever_config.get("limit_per_venue", 1000))

        for venue_name, venue_ids in self._iter_venue_ids():
            for venue_id in venue_ids:
                notes = self._get_openreview_notes(venue_id, limit)
                logger.info(f"Retrieved {len(notes)} notes for {venue_name} ({venue_id})")
                for note in notes:
                    note_id = note.get("id")
                    if not note_id or note_id in seen_note_ids:
                        continue
                    seen_note_ids.add(note_id)
                    raw_papers.append(OpenReviewPaperItem(note=note, venue=venue_name))

        raw_papers.sort(key=lambda item: self._note_sort_timestamp(item.note), reverse=True)
        if self.config.executor.debug:
            raw_papers = raw_papers[:10]
        return raw_papers

    def _iter_venue_ids(self) -> list[tuple[str, list[str]]]:
        year = str(self.retriever_config.get("year"))
        venue_id_patterns = dict(self.retriever_config.get("venue_id_patterns", {}))
        venue_ids = dict(self.retriever_config.get("venue_ids", {}))
        aliases = dict(self.retriever_config.get("aliases", {}))

        result: list[tuple[str, list[str]]] = []
        for venue in self.retriever_config.venues:
            venue_key = aliases.get(str(venue).lower(), str(venue).lower())
            configured_ids = venue_ids.get(venue_key)
            if configured_ids is None:
                pattern = venue_id_patterns.get(venue_key)
                if pattern is None:
                    raise ValueError(f"No OpenReview venue ID pattern configured for {venue}")
                configured_ids = [pattern.format(year=year)]
            elif isinstance(configured_ids, str):
                configured_ids = [configured_ids]
            result.append((venue_key, list(configured_ids)))
        return result

    def _get_openreview_notes(self, venue_id: str, limit: int) -> list[dict[str, Any]]:
        for api_url in (OPENREVIEW_API2_NOTES_URL, OPENREVIEW_API1_NOTES_URL):
            try:
                notes = self._fetch_all_notes(api_url, venue_id, limit)
            except (OSError, URLError, TimeoutError, json.JSONDecodeError) as exc:
                logger.warning(f"Failed to fetch {venue_id} from {api_url}: {exc}")
                continue
            if notes:
                return notes
        return []

    def _fetch_all_notes(self, api_url: str, venue_id: str, limit: int) -> list[dict[str, Any]]:
        notes: list[dict[str, Any]] = []
        offset = 0
        batch_size = min(limit, int(self.retriever_config.get("batch_size", 1000)))
        timeout = int(self.retriever_config.get("timeout_seconds", 30))

        while offset < limit:
            params = urlencode({"content.venueid": venue_id, "limit": batch_size, "offset": offset})
            request = Request(
                f"{api_url}?{params}",
                headers={"User-Agent": "zotero-arxiv-daily/1.0"},
            )
            with urlopen(request, timeout=timeout) as response:
                payload = json.load(response)
            batch = payload.get("notes", [])
            notes.extend(batch)
            if len(batch) < batch_size:
                break
            offset += batch_size
        return notes

    def convert_to_paper(self, raw_paper: OpenReviewPaperItem) -> Paper | None:
        note = raw_paper.note
        content = note.get("content", {})
        title = self._content_value(content, "title")
        abstract = self._content_value(content, "abstract") or ""
        if not title:
            logger.warning(f"Skipping OpenReview note without title: {note.get('id')}")
            return None

        authors = self._content_value(content, "authors") or []
        if isinstance(authors, str):
            authors = [authors]
        authors = [str(author) for author in authors]

        note_id = note.get("id")
        url = OPENREVIEW_FORUM_URL.format(id=note_id) if note_id else "https://openreview.net"
        pdf_url = self._pdf_url(note)
        venue = self._content_value(content, "venue") or raw_paper.venue.upper()
        full_text = f"Venue: {venue}\n\nAbstract: {abstract}" if abstract else f"Venue: {venue}"

        return Paper(
            source=self.name,
            title=title,
            authors=authors,
            abstract=abstract,
            url=url,
            pdf_url=pdf_url,
            full_text=full_text,
        )

    @staticmethod
    def _content_value(content: dict[str, Any], key: str) -> Any:
        value = content.get(key)
        if isinstance(value, dict) and "value" in value:
            return value["value"]
        return value

    def _pdf_url(self, note: dict[str, Any]) -> str | None:
        content = note.get("content", {})
        pdf = self._content_value(content, "pdf")
        if isinstance(pdf, str) and pdf.startswith("http"):
            return pdf
        note_id = note.get("id")
        return OPENREVIEW_PDF_URL.format(id=note_id) if note_id else None

    @staticmethod
    def _note_sort_timestamp(note: dict[str, Any]) -> int:
        for key in ("pdate", "tcdate", "tmdate", "cdate", "mdate"):
            value = note.get(key)
            if isinstance(value, int):
                return value
        return 0
