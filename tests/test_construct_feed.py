from datetime import datetime, timezone
import xml.etree.ElementTree as ET

from omegaconf import OmegaConf
import pytest

from tests.canned_responses import make_sample_paper
from zotero_arxiv_daily.construct_feed import ATOM, ZAD, write_atom_feed


def atom_config(path):
    return OmegaConf.create(
        {
            "path": str(path),
            "title": "Astronomy Daily",
            "description": "Curated astronomy papers",
            "site_url": "https://example.github.io/project/",
            "feed_url": "https://example.github.io/project/index.xml",
        }
    )


def test_atom_feed_contains_open_content_summary(tmp_path):
    paper = make_sample_paper(
        external_ids={"arxiv": "2608.01234"},
        published_at=datetime(2026, 8, 22, tzinfo=timezone.utc),
        tldr="Open-content TLDR",
        translated_title="翻译标题",
        journal="arXiv",
        score=7.25,
    )
    output = write_atom_feed([paper], atom_config(tmp_path / "index.xml"))
    root = ET.parse(output).getroot()
    entry = root.find(f"{{{ATOM}}}entry")
    assert entry.find(f"{{{ATOM}}}id").text == "arxiv:2608.01234"
    assert entry.find(f"{{{ATOM}}}summary").text == "Open-content TLDR"
    assert entry.find(f"{{{ZAD}}}translatedTitle").text == "翻译标题"


def test_scix_only_record_is_excluded_from_public_feed_by_default(tmp_path):
    paper = make_sample_paper(
        source="scix",
        abstract="LICENSED SCIX ABSTRACT SENTINEL",
        external_ids={"ads": "2026ApJ...999....1A"},
        url="https://scixplorer.org/abs/2026ApJ...999....1A/abstract",
        pdf_url=None,
        tldr=None,
        content_source="scix",
        remote_processing_allowed=False,
    )
    output = write_atom_feed([paper], atom_config(tmp_path / "index.xml"))
    xml = output.read_text()
    assert "LICENSED SCIX ABSTRACT SENTINEL" not in xml
    assert "2026ApJ...999....1A" not in xml
    assert ET.parse(output).getroot().find(f"{{{ATOM}}}entry") is None


def test_arxiv_enriched_scix_record_is_allowed_in_public_feed(tmp_path):
    paper = make_sample_paper(
        source="scix",
        external_ids={"ads": "2026ApJ...999....1A", "arxiv": "2608.01234"},
        url="https://scixplorer.org/abs/2026ApJ...999....1A/abstract",
        content_source="arxiv",
        remote_processing_allowed=True,
    )
    output = write_atom_feed([paper], atom_config(tmp_path / "index.xml"))
    entry = ET.parse(output).getroot().find(f"{{{ATOM}}}entry")
    assert entry is not None
    assert entry.find(f"{{{ATOM}}}id").text == "ads:2026ApJ...999....1A"


def test_scix_metadata_requires_explicit_include_policy(tmp_path):
    paper = make_sample_paper(
        source="scix",
        abstract="LICENSED SCIX ABSTRACT SENTINEL",
        external_ids={"ads": "2026ApJ...999....1A"},
        content_source="scix",
        remote_processing_allowed=False,
    )
    config = atom_config(tmp_path / "index.xml")
    config.scix_only_policy = "include"
    output = write_atom_feed([paper], config)
    xml = output.read_text()
    assert "2026ApJ...999....1A" in xml
    assert "LICENSED SCIX ABSTRACT SENTINEL" not in xml


def test_invalid_scix_feed_policy_fails_closed(tmp_path):
    config = atom_config(tmp_path / "index.xml")
    config.scix_only_policy = "typo"
    with pytest.raises(ValueError, match="scix_only_policy"):
        write_atom_feed([], config)
