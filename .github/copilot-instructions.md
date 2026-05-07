# Copilot Instructions

## Project Overview

Zotero-arXiv-Daily recommends new arXiv/bioRxiv/medRxiv papers based on a user's Zotero library. It computes embedding similarity between new papers and the user's existing library, generates TLDRs via LLM, and delivers results by email. Designed to run as a GitHub Actions workflow at zero cost.

## Commands

```bash
# Install/sync dependencies
uv sync

# Run the application
uv run src/zotero_arxiv_daily/main.py

# Run tests (excludes slow tests by default)
uv run pytest

# Run all tests including slow ones
uv run pytest -m ""

# Run a single test
uv run pytest tests/test_utils.py::TestGlobMatch -v
```

No linter or formatter is configured.

## Architecture

The app is a linear pipeline orchestrated by `Executor` (`src/zotero_arxiv_daily/executor.py`):

1. **Fetch Zotero corpus** → pyzotero API
2. **Filter corpus** → `include_path` / `ignore_path` glob patterns
3. **Retrieve new papers** → from configured sources (arXiv RSS, bioRxiv/medRxiv REST API)
4. **Rerank** → weighted embedding similarity to corpus (newer Zotero papers weighted higher)
5. **Generate TLDRs + affiliations** → OpenAI-compatible LLM API
6. **Render + send email** → HTML email via SMTP

### Plugin Systems

**Retrievers** (`src/zotero_arxiv_daily/retriever/`): Register via `@register_retriever("name")` decorator on a `BaseRetriever` subclass. Each retriever implements `_retrieve_raw_papers()` and `convert_to_paper()`. Discovered at runtime via `get_retriever_cls(name)` from a module-level `registered_retrievers` dict.

**Rerankers** (`src/zotero_arxiv_daily/reranker/`): Register via `@register_reranker("name")` decorator on a `BaseReranker` subclass. Two implementations: `local` (sentence-transformers) and `api` (OpenAI-compatible embeddings endpoint). Discovered via `get_reranker_cls(name)`.

When adding a new retriever or reranker, follow the existing pattern: create a new file, subclass the base, apply the registration decorator, and implement the abstract methods.

### Configuration

Uses Hydra + OmegaConf. Config composes from `config/base.yaml` (defaults with `???` placeholders for required values) + `config/custom.yaml` (user overrides). The composition order is defined in `config/default.yaml`. Environment variables are interpolated via `${oc.env:VAR_NAME,default}` syntax. Entry point uses `@hydra.main(config_name="default")`.

### Data Classes

`Paper` and `CorpusPaper` in `src/zotero_arxiv_daily/protocol.py`. `Paper` has LLM-powered methods (`generate_tldr`, `generate_affiliations`) that call the OpenAI API directly with `tiktoken`-based token truncation.

## Testing Conventions

- Tests use **pytest monkeypatch + `SimpleNamespace`** for stubs — not `unittest.mock`.
- A session-scoped Hydra config in `tests/conftest.py` is deep-copied per test via the `config` fixture.
- Canned response factories live in `tests/canned_responses.py` (e.g., `make_stub_openai_client()`, `make_stub_zotero_client()`).
- Tests marked `@pytest.mark.slow` require heavy dependencies (model downloads) and are excluded by default (`addopts = "-m 'not slow'"` in pyproject.toml).
- Monkeypatching targets the module-level import path (e.g., `"zotero_arxiv_daily.executor.zotero.Zotero"`).

## Coding Conventions

- **Logging:** `loguru.logger` throughout — never `print()` or stdlib `logging`.
- **Type hints:** Modern Python 3.10+ syntax (`list[Paper]`, `str | None`).
- **Constants:** Module-level `UPPER_SNAKE_CASE`.
- **Private methods:** Prefixed with `_` (e.g., `_retrieve_raw_papers`).
- **Error handling:** Graceful degradation with try/except and fallback logic; log warnings rather than raising.
- **Config injection:** All major components receive `DictConfig` at init and store it as `self.config`.

## Git Workflow

- PRs should target the **`dev`** branch, not `main`.
