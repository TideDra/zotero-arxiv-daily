from datetime import datetime, timezone
import os

from omegaconf import OmegaConf
import pytest

from zotero_arxiv_daily.retriever.ads_retriever import AdsRetriever, ScixRetriever, extract_arxiv_id
from zotero_arxiv_daily.retriever.base import registered_retrievers


class StubResponse:
    def __init__(self, payload, status_code=200, headers=None):
        self._payload = payload
        self.status_code = status_code
        self.headers = headers or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def make_config(source="scix", **source_overrides):
    source_config = {
        "api_token": "scix-secret",
        "api_base": "https://api.adsabs.harvard.edu/v1",
        "query": 'database:astronomy abs:"Milky Way"',
        "lookback_days": 7,
        "max_results": 200,
    }
    source_config.update(source_overrides)
    return OmegaConf.create({"source": {source: source_config}, "executor": {"debug": False}})


def sample_doc():
    return {
        "bibcode": "2026ApJ...999....1A",
        "title": ["A Galactic Dynamics Paper"],
        "author": ["Astronomer, A.", "Researcher, B."],
        "abstract": "A locally processed abstract.",
        "identifier": ["2026ApJ...999....1A", "arXiv:2608.01234", "10.1234/example"],
        "doi": ["10.1234/example"],
        "entry_date": "2026-08-22T10:00:00Z",
        "date": "2026-08-01T00:00:00Z",
        "pub": "The Astrophysical Journal",
        "keyword": ["Galaxy dynamics", "Milky Way"],
    }


def test_scix_request_and_conversion(monkeypatch):
    retriever = ScixRetriever(make_config())
    captured = {}

    def fake_get(url, params, headers, timeout):
        captured.update(url=url, params=params, headers=headers, timeout=timeout)
        return StubResponse(
            {"response": {"numFound": 1, "docs": [sample_doc()]}},
            headers={"X-RateLimit-Remaining": "4999", "X-RateLimit-Limit": "5000"},
        )

    monkeypatch.setattr(retriever.session, "get", fake_get)
    papers = retriever.retrieve_papers()

    assert len(papers) == 1
    paper = papers[0]
    assert paper.external_ids == {
        "ads": "2026ApJ...999....1A",
        "arxiv": "2608.01234",
        "doi": "10.1234/example",
    }
    assert paper.entry_at == datetime(2026, 8, 22, 10, 0, tzinfo=timezone.utc)
    assert paper.remote_processing_allowed is False
    assert paper.source == "scix"
    assert paper.stable_id == "ads:2026ApJ...999....1A"
    assert paper.url == "https://scixplorer.org/abs/2026ApJ...999....1A/abstract"
    assert captured["url"] == "https://api.adsabs.harvard.edu/v1/search/query"
    assert "entdate:[NOW-7DAYS TO *]" in captured["params"]["q"]
    assert captured["params"]["rows"] == 200
    assert captured["params"]["sort"] == "entry_date asc"
    assert {"identifier", "entry_date", "keyword_schema"}.issubset(
        set(captured["params"]["fl"].split(","))
    )
    assert captured["headers"]["Authorization"] == "Bearer scix-secret"


def test_scix_debug_request_is_limited_to_five_rows(monkeypatch):
    config = make_config()
    config.executor.debug = True
    retriever = ScixRetriever(config)
    captured = {}

    def fake_get(url, params, headers, timeout):
        captured.update(params)
        return StubResponse({"response": {"numFound": 1, "docs": [sample_doc()]}})

    monkeypatch.setattr(retriever.session, "get", fake_get)
    retriever.retrieve_papers()

    assert captured["rows"] == 5


def test_doi_does_not_become_arxiv_identifier():
    assert extract_arxiv_id(["10.1234/2608.01234"]) is None


def test_scix_rejects_bulk_result_set(monkeypatch):
    retriever = ScixRetriever(make_config(max_results=10))
    monkeypatch.setattr(
        retriever.session,
        "get",
        lambda *args, **kwargs: StubResponse({"response": {"numFound": 11, "docs": []}}),
    )
    with pytest.raises(RuntimeError, match="Narrow source.scix.query"):
        retriever.retrieve_papers()


def test_scix_authentication_error_is_actionable(monkeypatch):
    retriever = ScixRetriever(make_config())
    monkeypatch.setattr(
        retriever.session,
        "get",
        lambda *args, **kwargs: StubResponse({}, status_code=401),
    )
    with pytest.raises(RuntimeError, match="SciX API authentication failed"):
        retriever.retrieve_papers()


def test_scix_requires_token_and_query(monkeypatch):
    monkeypatch.delenv("SCIX_API_TOKEN", raising=False)
    monkeypatch.delenv("ADS_API_TOKEN", raising=False)
    with pytest.raises(ValueError, match="api_token"):
        ScixRetriever(make_config(api_token=None))
    with pytest.raises(ValueError, match="query"):
        ScixRetriever(make_config(query=None))


@pytest.mark.parametrize(
    ("environment", "expected"),
    [({"SCIX_API_TOKEN": "new-token", "ADS_API_TOKEN": "old-token"}, "new-token"),
     ({"ADS_API_TOKEN": "old-token"}, "old-token")],
)
def test_scix_token_environment_fallback(monkeypatch, environment, expected):
    monkeypatch.delenv("SCIX_API_TOKEN", raising=False)
    monkeypatch.delenv("ADS_API_TOKEN", raising=False)
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    assert ScixRetriever(make_config(api_token=None)).api_token == expected


def test_legacy_ads_alias_reads_legacy_config_and_preserves_state_id():
    retriever = AdsRetriever(make_config(source="ads", api_token="legacy-token"))
    paper = retriever.convert_to_paper(sample_doc())
    assert retriever.api_token == "legacy-token"
    assert paper.source == "ads"
    assert paper.stable_id == "ads:2026ApJ...999....1A"
    assert registered_retrievers["scix"] is ScixRetriever
    assert registered_retrievers["ads"] is AdsRetriever
    assert registered_retrievers["scix"] is not registered_retrievers["ads"]


def test_scix_conversion_accepts_scalar_identifier_author_and_keyword():
    retriever = ScixRetriever(make_config())
    raw = sample_doc()
    raw.update(identifier="arXiv:2608.01234", author="Solo, A.", keyword="Milky Way")
    paper = retriever.convert_to_paper(raw)
    assert paper.external_ids["arxiv"] == "2608.01234"
    assert paper.authors == ["Solo, A."]
    assert paper.keywords == ["Milky Way"]


@pytest.mark.live
def test_scix_live_search_contract():
    if os.getenv("SCIX_LIVE_TEST") != "1":
        pytest.skip("Set SCIX_LIVE_TEST=1 to opt in to the one-row live SciX contract test.")
    token = os.getenv("SCIX_API_TOKEN") or os.getenv("ADS_API_TOKEN")
    if not token:
        pytest.skip("SCIX_API_TOKEN or ADS_API_TOKEN is required for the live contract test.")
    config = make_config(api_token=token, query="database:astronomy", max_results=1, lookback_days=30)
    payload = ScixRetriever(config)._request()
    docs = payload["response"]["docs"]
    assert len(docs) <= 1
    if docs:
        assert "bibcode" in docs[0]
        assert "identifier" in docs[0]
