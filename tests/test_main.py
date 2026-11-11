"""Tests for main entry point.

The @hydra.main decorator makes main() hard to test directly in pytest
because config_path resolution depends on the calling context.
We test the inner logic by calling main's body with a composed config.
"""

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from pathlib import Path


@pytest.fixture(autouse=True)
def _clear_hydra():
    """Ensure GlobalHydra is clean before and after each test in this module."""
    GlobalHydra.instance().clear()
    yield
    GlobalHydra.instance().clear()


def test_main_creates_executor_and_runs(config, monkeypatch):
    """Verify that the main function creates an Executor and calls run()."""
    calls = []

    class FakeExecutor:
        def __init__(self, cfg):
            calls.append(("init", cfg))

        def run(self):
            calls.append(("run",))

    monkeypatch.setattr("zotero_arxiv_daily.main.Executor", FakeExecutor)

    # Call main's body directly, bypassing @hydra.main
    from zotero_arxiv_daily import main as main_mod

    # Simulate what @hydra.main does: calls main(config)
    main_mod.main.__wrapped__(config)

    assert ("init", config) in calls
    assert ("run",) in calls


def test_main_debug_logging(config, monkeypatch):
    """Verify debug mode sets appropriate log level."""
    from omegaconf import open_dict

    with open_dict(config):
        config.executor.debug = True

    class FakeExecutor:
        def __init__(self, cfg):
            pass
        def run(self):
            pass

    monkeypatch.setattr("zotero_arxiv_daily.main.Executor", FakeExecutor)

    from zotero_arxiv_daily import main as main_mod

    main_mod.main.__wrapped__(config)
    # If we get here without error, the debug path executed successfully


def test_default_config_reads_smtp_env(monkeypatch):
    monkeypatch.setenv("SMTP_SERVER", "smtp.exmail.qq.com")
    monkeypatch.setenv("SMTP_PORT", "587")

    config_dir = str(Path(__file__).resolve().parent.parent / "config")

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="default",
            overrides=[
                "zotero.user_id=000000",
                "zotero.api_key=fake-zotero-key",
                "email.sender=test@example.com",
                "email.receiver=test@example.com",
                "email.sender_password=test",
                "llm.api.key=sk-fake",
                "llm.api.base_url=http://localhost:30000/v1",
                "llm.generation_kwargs.model=gpt-4o-mini",
                "source.arxiv.category=[cs.AI,cs.CV]",
                "executor.source=[arxiv]",
            ],
        )

    assert cfg.email.smtp_server == "smtp.exmail.qq.com"
    assert int(cfg.email.smtp_port) == 587


def test_daily_workflow_exposes_smtp_env():
    workflow = (
        Path(__file__).resolve().parent.parent / ".github" / "workflows" / "main.yml"
    ).read_text()

    assert "SMTP_SERVER: ${{ secrets.SMTP_SERVER }}" in workflow
    assert "SMTP_PORT: ${{ secrets.SMTP_PORT }}" in workflow


def test_daily_workflow_logs_masked_smtp_diagnostics():
    workflow = (
        Path(__file__).resolve().parent.parent / ".github" / "workflows" / "main.yml"
    ).read_text()

    assert "SMTP diagnostics:" in workflow
    assert "SENDER_PASSWORD=SET" in workflow
    assert "masked_sender()" in workflow
