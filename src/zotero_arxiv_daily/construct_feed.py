from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET

from omegaconf import DictConfig

from .protocol import Paper


ATOM = "http://www.w3.org/2005/Atom"
ZAD = "https://github.com/TideDra/zotero-arxiv-daily/ns"
ET.register_namespace("", ATOM)
ET.register_namespace("zad", ZAD)


def _atom(tag: str) -> str:
    return f"{{{ATOM}}}{tag}"


def _zad(tag: str) -> str:
    return f"{{{ZAD}}}{tag}"


def _format_datetime(value: datetime | None, fallback: datetime) -> str:
    value = value or fallback
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def select_public_papers(papers: list[Paper], atom_config: DictConfig) -> list[Paper]:
    """Apply the fail-closed SciX redistribution policy before serializing entries."""
    policy = str(atom_config.get("scix_only_policy", "exclude")).lower()
    if policy not in {"exclude", "include"}:
        raise ValueError("output.atom.scix_only_policy must be 'exclude' or 'include'.")
    if policy == "include":
        return papers
    return [
        paper
        for paper in papers
        if not ("ads" in paper.external_ids and paper.content_source != "arxiv")
    ]


def build_atom_tree(papers: list[Paper], atom_config: DictConfig) -> ET.ElementTree:
    now = datetime.now(timezone.utc)
    papers = select_public_papers(papers, atom_config)
    feed = ET.Element(_atom("feed"))
    ET.SubElement(feed, _atom("id")).text = str(atom_config.feed_url or atom_config.site_url)
    ET.SubElement(feed, _atom("title")).text = str(atom_config.title)
    ET.SubElement(feed, _atom("subtitle")).text = str(atom_config.description)
    ET.SubElement(feed, _atom("updated")).text = _format_datetime(None, now)
    ET.SubElement(feed, _atom("link"), href=str(atom_config.site_url), rel="alternate")
    ET.SubElement(feed, _atom("link"), href=str(atom_config.feed_url), rel="self", type="application/atom+xml")

    for paper in papers:
        entry = ET.SubElement(feed, _atom("entry"))
        ET.SubElement(entry, _atom("id")).text = paper.stable_id
        ET.SubElement(entry, _atom("title")).text = paper.title
        ET.SubElement(entry, _atom("link"), href=paper.url, rel="alternate")
        if paper.pdf_url:
            ET.SubElement(entry, _atom("link"), href=paper.pdf_url, rel="related", type="application/pdf")
        timestamp = paper.entry_at or paper.published_at
        ET.SubElement(entry, _atom("updated")).text = _format_datetime(timestamp, now)
        if paper.published_at:
            ET.SubElement(entry, _atom("published")).text = _format_datetime(paper.published_at, now)
        for author in paper.authors:
            author_element = ET.SubElement(entry, _atom("author"))
            ET.SubElement(author_element, _atom("name")).text = author
        ET.SubElement(entry, _atom("category"), term=paper.source)
        if paper.journal:
            ET.SubElement(entry, _zad("journal")).text = paper.journal
        if paper.score is not None:
            ET.SubElement(entry, _zad("relevanceScore")).text = f"{paper.score:.6f}"
        if paper.translated_title:
            ET.SubElement(entry, _zad("translatedTitle")).text = paper.translated_title
        if paper.tldr:
            ET.SubElement(entry, _atom("summary")).text = paper.tldr

    return ET.ElementTree(feed)


def write_atom_feed(papers: list[Paper], atom_config: DictConfig) -> Path:
    output_path = Path(str(atom_config.path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree = build_atom_tree(papers, atom_config)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    ET.parse(output_path)
    return output_path
