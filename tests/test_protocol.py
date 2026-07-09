"""Tests for zotero_arxiv_daily.protocol: Paper.generate_tldr, Paper.generate_affiliations."""

from types import SimpleNamespace

import pytest

from tests.canned_responses import make_sample_paper, make_stub_openai_client


@pytest.fixture()
def llm_params():
    return {
        "language": "English",
        "generation_kwargs": {"model": "gpt-4o-mini", "max_tokens": 16384},
    }


def _client_returning(content: str):
    """A stub OpenAI client whose chat.completions.create() returns fixed content."""

    def create(**kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


# ---------------------------------------------------------------------------
# generate_tldr
# ---------------------------------------------------------------------------


def test_tldr_returns_response(llm_params):
    client = make_stub_openai_client()
    paper = make_sample_paper()
    result = paper.generate_tldr(client, llm_params)
    # Default 'structured' style: labelled HTML built from the stub's JSON.
    assert "<strong>Problem:</strong>" in result
    assert "Readers waste time re-reading dense paper abstracts." in result
    assert paper.tldr == result


def test_tldr_structured_format(llm_params):
    client = make_stub_openai_client()
    paper = make_sample_paper()
    result = paper.generate_tldr(client, llm_params)
    for label in ("<strong>Problem:</strong>", "<strong>Idea:</strong>", "<strong>Result:</strong>"):
        assert label in result
    assert "<br>" in result


def test_tldr_result_optional(llm_params):
    """An empty 'result' value should drop the Result line, keeping Problem/Idea."""
    client = _client_returning('{"problem": "P text", "idea": "I text", "result": ""}')
    paper = make_sample_paper()
    result = paper.generate_tldr(client, llm_params)
    assert "<strong>Problem:</strong> P text" in result
    assert "<strong>Idea:</strong> I text" in result
    assert "Result:" not in result


def test_tldr_custom_labels(llm_params):
    """Labels from config override the English defaults."""
    llm_params = {**llm_params, "tldr": {"labels": {"problem": "问题", "idea": "思路"}}}
    client = make_stub_openai_client()
    paper = make_sample_paper()
    result = paper.generate_tldr(client, llm_params)
    assert "<strong>问题:</strong>" in result
    assert "<strong>思路:</strong>" in result


def test_tldr_plain_style(llm_params):
    """style='plain' returns the raw one-liner without JSON formatting."""
    llm_params = {**llm_params, "tldr": {"style": "plain"}}
    client = _client_returning("A concise one-liner summary.")
    paper = make_sample_paper()
    result = paper.generate_tldr(client, llm_params)
    assert result == "A concise one-liner summary."


def test_tldr_malformed_json_falls_back(llm_params):
    """Structured style with non-JSON output falls back to the abstract."""
    client = _client_returning("Sorry, I could not produce JSON.")
    paper = make_sample_paper()
    result = paper.generate_tldr(client, llm_params)
    assert result == paper.abstract


def test_tldr_without_abstract_or_fulltext(llm_params):
    client = make_stub_openai_client()
    paper = make_sample_paper(abstract="", full_text=None)
    result = paper.generate_tldr(client, llm_params)
    assert "Failed to generate TLDR" in result


def test_tldr_falls_back_to_abstract_on_error(llm_params):
    paper = make_sample_paper()

    # Client whose create() raises
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
