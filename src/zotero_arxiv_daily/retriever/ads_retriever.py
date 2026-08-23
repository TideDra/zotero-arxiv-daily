"""Compatibility imports for integrations that still reference ads_retriever."""

from .scix_retriever import (
    AdsRetriever,
    ARXIV_ID_PATTERN,
    SCIX_API_BASE,
    SCIX_FIELDS,
    SCIX_RECORD_BASE,
    ScixRetriever,
    extract_arxiv_id,
)

ADS_FIELDS = SCIX_FIELDS

__all__ = [
    "ADS_FIELDS",
    "AdsRetriever",
    "ARXIV_ID_PATTERN",
    "SCIX_API_BASE",
    "SCIX_FIELDS",
    "SCIX_RECORD_BASE",
    "ScixRetriever",
    "extract_arxiv_id",
]
