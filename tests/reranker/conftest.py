"""Reranker-specific fixtures."""

import pytest

from tests.canned_responses import make_stub_openai_client


@pytest.fixture()
def patch_openai(monkeypatch):
    """Patch OpenAI constructor where ApiReranker imports it.

    ApiReranker does ``from openai import OpenAI`` at module level, so we
    must patch the name in *its* namespace.
    """
    stub = make_stub_openai_client()
    monkeypatch.setattr("zotero_arxiv_daily.reranker.api.OpenAI", lambda **kwargs: stub)
    return stub
