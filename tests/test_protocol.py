"""Tests for zotero_arxiv_daily.protocol: Paper.generate_tldr, Paper.generate_affiliations."""

import pytest

from tests.canned_responses import make_sample_paper, make_stub_openai_client


@pytest.fixture()
def llm_params():
    return {
        "language": "English",
        "generation_kwargs": {"model": "gpt-4o-mini", "max_tokens": 16384},
    }


# ---------------------------------------------------------------------------
# generate_tldr
# ---------------------------------------------------------------------------


def test_tldr_returns_response(llm_params):
    client = make_stub_openai_client()
    paper = make_sample_paper()
    result = paper.generate_tldr(client, llm_params)
    assert result == "Hello! How can I assist you today?"
    assert paper.tldr == result


def test_tldr_without_abstract_or_fulltext(llm_params):
    client = make_stub_openai_client()
    paper = make_sample_paper(abstract="", full_text=None)
    result = paper.generate_tldr(client, llm_params)
    assert "Failed to generate TLDR" in result


def test_tldr_falls_back_to_abstract_on_error(llm_params):
    paper = make_sample_paper()

    # Client whose create() raises
    from types import SimpleNamespace

    broken_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kw: (_ for _ in ()).throw(RuntimeError("API down")))
        )
    )
    result = paper.generate_tldr(broken_client, llm_params)
    assert result == paper.abstract


def test_tldr_truncates_long_prompt(llm_params):
    client = make_stub_openai_client()
    paper = make_sample_paper(full_text="word " * 10000)
    result = paper.generate_tldr(client, llm_params)
    assert result is not None


# ---------------------------------------------------------------------------
# generate_affiliations
# ---------------------------------------------------------------------------


def test_affiliations_returns_parsed_list(llm_params):
    client = make_stub_openai_client()
    paper = make_sample_paper()
    result = paper.generate_affiliations(client, llm_params)
    assert isinstance(result, list)
    assert "TsingHua University" in result
    assert "Peking University" in result


def test_affiliations_none_without_fulltext(llm_params):
    client = make_stub_openai_client()
    paper = make_sample_paper(full_text=None)
    result = paper.generate_affiliations(client, llm_params)
    assert result is None


def test_affiliations_deduplicates(llm_params):
    """The stub returns two distinct affiliations, so no dedup needed.
    But confirm the set() dedup in the code doesn't break anything.
    """
    client = make_stub_openai_client()
    paper = make_sample_paper()
    result = paper.generate_affiliations(client, llm_params)
    assert len(result) == len(set(result))


def test_affiliations_malformed_llm_output(llm_params):
    """LLM returns affiliations without JSON brackets. Should fall back gracefully."""
    from types import SimpleNamespace

    def create_no_brackets(**kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="TsingHua University, Peking University"),
                )
            ]
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create_no_brackets)
        )
    )
    paper = make_sample_paper()
    result = paper.generate_affiliations(client, llm_params)
    # re.search for [...] will fail -> AttributeError -> caught -> returns None
    assert result is None


def test_affiliations_error_returns_none(llm_params):
    from types import SimpleNamespace

    broken_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kw: (_ for _ in ()).throw(RuntimeError("boom")))
        )
    )
    paper = make_sample_paper()
    result = paper.generate_affiliations(broken_client, llm_params)
    assert result is None
    assert paper.affiliations is None


# ---------------------------------------------------------------------------
# LiteLLM backend (llm.use_litellm)
# ---------------------------------------------------------------------------


_AFFIL_MARKER = "You are an assistant who perfectly extracts affiliations"


def _make_litellm_stub(recorder):
    """Stand-in for the litellm module: records completion kwargs and returns
    an OpenAI-shaped response."""
    from types import SimpleNamespace

    def completion(**kwargs):
        recorder.append(kwargs)
        content = (
            '["TsingHua University","Peking University"]'
            if _AFFIL_MARKER in str(kwargs.get("messages", []))
            else "A one-sentence summary."
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    return SimpleNamespace(completion=completion)


def _litellm_params(**overrides):
    params = {
        "use_litellm": True,
        "language": "English",
        "api": {"key": "secret", "base_url": "http://localhost:4000"},
        "generation_kwargs": {"model": "anthropic/claude-opus-4-8", "max_tokens": 512},
    }
    params.update(overrides)
    return params


def test_litellm_tldr_dispatch(monkeypatch):
    import zotero_arxiv_daily.protocol as protocol

    recorder = []
    monkeypatch.setattr(protocol, "litellm", _make_litellm_stub(recorder))

    result = make_sample_paper().generate_tldr(None, _litellm_params())

    assert result == "A one-sentence summary."
    assert len(recorder) == 1
    call = recorder[0]
    assert call["model"] == "anthropic/claude-opus-4-8"
    assert call["drop_params"] is True
    assert call["api_key"] == "secret"
    assert call["api_base"] == "http://localhost:4000"


def test_litellm_omits_blank_credentials(monkeypatch):
    import zotero_arxiv_daily.protocol as protocol

    recorder = []
    monkeypatch.setattr(protocol, "litellm", _make_litellm_stub(recorder))

    make_sample_paper().generate_tldr(None, _litellm_params(api={"key": None, "base_url": ""}))

    call = recorder[0]
    assert "api_key" not in call  # -> LiteLLM falls back to the provider env var
    assert "api_base" not in call


def test_litellm_affiliations(monkeypatch):
    import zotero_arxiv_daily.protocol as protocol

    recorder = []
    monkeypatch.setattr(protocol, "litellm", _make_litellm_stub(recorder))

    result = make_sample_paper().generate_affiliations(None, _litellm_params())
    assert "TsingHua University" in result
    assert "Peking University" in result


def test_litellm_missing_dependency_falls_back(monkeypatch):
    import zotero_arxiv_daily.protocol as protocol

    # Simulate litellm not installed while use_litellm is set.
    monkeypatch.setattr(protocol, "litellm", None)
    paper = make_sample_paper()
    result = paper.generate_tldr(None, _litellm_params())
    # generate_tldr swallows the ImportError and falls back to the abstract.
    assert result == paper.abstract
