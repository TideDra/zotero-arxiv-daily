from __future__ import annotations

from datetime import datetime
import os
import re
from time import sleep
from typing import Any

from loguru import logger
import requests

from .base import BaseRetriever, register_retriever
from ..protocol import Paper
from .arxiv_retriever import canonical_arxiv_id


SCIX_API_BASE = "https://api.adsabs.harvard.edu/v1"
SCIX_RECORD_BASE = "https://scixplorer.org/abs"
SCIX_FIELDS = (
    "bibcode",
    "title",
    "author",
    "abstract",
    "identifier",
    "doi",
    "date",
    "entry_date",
    "pubdate",
    "pub",
    "doctype",
    "property",
    "keyword",
    "keyword_schema",
    "arxiv_class",
)
ARXIV_ID_PATTERN = re.compile(
    r"(?:arXiv:)?(?P<identifier>(?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Za-z-]+)?/\d{7})(?:v\d+)?)"
)


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        try:
            return datetime.strptime(value, "%Y-%m-%d")
        except ValueError:
            logger.warning(f"Ignoring invalid SciX/ADS date: {value}")
            return None


def _first(value: Any, default: str = "") -> str:
    if isinstance(value, list):
        return str(value[0]) if value else default
    return str(value) if value is not None else default


def _as_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    return [str(value)]


def extract_arxiv_id(identifiers: list[str] | None) -> str | None:
    for identifier in identifiers or []:
        match = ARXIV_ID_PATTERN.fullmatch(identifier)
        if match is None and "arxiv" in identifier.lower():
            match = ARXIV_ID_PATTERN.search(identifier)
        if match:
            return canonical_arxiv_id(match.group("identifier"))
    return None


@register_retriever("scix")
class ScixRetriever(BaseRetriever):
    """Retrieve a bounded astronomy candidate set through the SciX/ADS API."""

    def __init__(self, config):
        super().__init__(config)
        configured_token = self.retriever_config.get("api_token")
        self.api_token = configured_token or os.getenv("SCIX_API_TOKEN") or os.getenv("ADS_API_TOKEN")
        self.api_base = str(self.retriever_config.get("api_base", SCIX_API_BASE)).rstrip("/")
        self.query = self.retriever_config.get("query")
        self.lookback_days = int(self.retriever_config.get("lookback_days", 7))
        self.max_results = int(self.retriever_config.get("max_results", 200))
        self.session = requests.Session()
        config_key = f"source.{self.name}"
        if not self.api_token:
            raise ValueError(
                f"{config_key}.api_token must be set; use SCIX_API_TOKEN "
                "or the legacy ADS_API_TOKEN fallback."
            )
        if not self.query:
            raise ValueError(f"{config_key}.query must contain a SciX search query.")
        if not 1 <= self.max_results <= 2000:
            raise ValueError(f"{config_key}.max_results must be between 1 and 2000.")
        if self.lookback_days < 1:
            raise ValueError(f"{config_key}.lookback_days must be at least 1.")

    def _request(self) -> dict[str, Any]:
        endpoint = f"{self.api_base}/search/query"
        query = f"({self.query}) entdate:[NOW-{self.lookback_days}DAYS TO *]"
        params = {
            "q": query,
            "fl": ",".join(SCIX_FIELDS),
            "rows": min(self.max_results, 5) if self.config.executor.debug else self.max_results,
            "sort": "entry_date asc",
        }
        headers = {"Authorization": f"Bearer {self.api_token}"}
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = self.session.get(endpoint, params=params, headers=headers, timeout=(10, 60))
            except requests.RequestException as exc:
                last_error = exc
                if attempt == 2:
                    raise RuntimeError("SciX API request failed after 3 attempts") from exc
                sleep(2 ** attempt)
                continue

            remaining = response.headers.get("X-RateLimit-Remaining")
            limit = response.headers.get("X-RateLimit-Limit")
            reset = response.headers.get("X-RateLimit-Reset")
            if remaining is not None:
                logger.info(f"SciX/ADS API rate limit: {remaining}/{limit} remaining; reset={reset}")
            if response.status_code in {401, 403}:
                raise RuntimeError(
                    "SciX API authentication failed; check SCIX_API_TOKEN or ADS_API_TOKEN."
                )
            if response.status_code == 429:
                raise RuntimeError(f"SciX API rate limit exhausted; reset={reset or 'unknown'}.")
            if response.status_code >= 500:
                last_error = RuntimeError(f"SciX API server returned HTTP {response.status_code}")
                if attempt == 2:
                    raise last_error
                sleep(2 ** attempt)
                continue
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise RuntimeError("SciX API returned a non-object response.")
            return payload
        raise RuntimeError("SciX API request failed") from last_error

    def _retrieve_raw_papers(self) -> list[dict[str, Any]]:
        payload = self._request()
        response = payload.get("response")
        if not isinstance(response, dict):
            raise RuntimeError("SciX API response is missing the response object.")
        num_found = int(response.get("numFound", 0))
        if num_found > self.max_results:
            raise RuntimeError(
                f"SciX query matched {num_found} records, above max_results={self.max_results}. "
                f"Narrow source.{self.name}.query instead of bulk-downloading results."
            )
        docs = response.get("docs", [])
        if not isinstance(docs, list):
            raise RuntimeError("SciX API response.docs must be a list.")
        return docs

    def convert_to_paper(self, raw_paper: dict[str, Any]) -> Paper:
        bibcode = str(raw_paper["bibcode"])
        identifiers = _as_strings(raw_paper.get("identifier"))
        # Keep the historical namespace so existing ads:<bibcode> state remains valid.
        external_ids = {"ads": bibcode}
        if arxiv_id := extract_arxiv_id(identifiers):
            external_ids["arxiv"] = arxiv_id
        doi = _first(raw_paper.get("doi"))
        if doi:
            external_ids["doi"] = doi
        return Paper(
            source=self.name,
            title=_first(raw_paper.get("title"), "Untitled SciX record"),
            authors=_as_strings(raw_paper.get("author")),
            abstract=str(raw_paper.get("abstract") or ""),
            url=f"{SCIX_RECORD_BASE}/{bibcode}/abstract",
            external_ids=external_ids,
            entry_at=_parse_datetime(raw_paper.get("entry_date")),
            published_at=_parse_datetime(raw_paper.get("date")),
            journal=str(raw_paper.get("pub") or "") or None,
            keywords=_as_strings(raw_paper.get("keyword")),
            content_source=self.name,
            remote_processing_allowed=False,
        )

    def retrieve_papers(self) -> list[Paper]:
        """SciX metadata conversion is local and should not inherit per-paper sleeps."""
        papers: list[Paper] = []
        for raw_paper in self._retrieve_raw_papers():
            try:
                papers.append(self.convert_to_paper(raw_paper))
            except Exception as exc:
                logger.warning(f"Skipping malformed SciX record {raw_paper.get('bibcode')}: {exc}")
        return papers


@register_retriever("ads")
class AdsRetriever(ScixRetriever):
    """Backward-compatible source.ads alias for pre-SciX configurations."""
