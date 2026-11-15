"""Build a daily slate from the 2026 CCF A AI conference catalogues."""

from concurrent.futures import ThreadPoolExecutor
from datetime import date
import gzip
from hashlib import sha256
import json
from unicodedata import normalize
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen
from xml.etree import ElementTree

from loguru import logger
from lxml import etree, html

from .protocol import Paper


DAILY_LIMIT = 20
CANDIDATE_LIMIT = 256
ROTATION_DAYS = 64
ACL_XML = "https://raw.githubusercontent.com/acl-org/acl-anthology/master/data/xml/2026.acl.xml"
CVPR_PROCEEDINGS = "https://openaccess.thecvf.com/CVPR2026?day=all"
ICML_PROCEEDINGS = "https://proceedings.mlr.press/v306/"
AAAI_FULL_PAPER_ISSUES = {str(issue) for issue in range(1, 46)}


def _get(url: str) -> bytes:
    request = Request(
        url,
        headers={
            "User-Agent": "zotero-arxiv-daily (https://github.com/Herren-Hsing/zotero-arxiv-daily)",
            "Accept-Encoding": "gzip",
        },
    )
    with urlopen(request, timeout=20) as response:
        body = response.read()
        if response.headers.get("Content-Encoding") == "gzip":
            return gzip.decompress(body)
        return body


def _title_key(title: str) -> str:
    title = normalize("NFKC", title).casefold()
    return "".join(char for char in title if char.isalnum())


def _xml_text(element: ElementTree.Element | None) -> str:
    return "".join(element.itertext()).strip() if element is not None else ""


def _acl_papers() -> list[Paper]:
    catalogue = ElementTree.fromstring(_get(ACL_XML))
    papers = []
    for entry in catalogue.findall("./volume[@id='long']/paper"):
        identifier = f"2026.acl-long.{entry.attrib['id']}"
        papers.append(Paper(
            source="ACL",
            title=_xml_text(entry.find("title")),
            authors=[
                " ".join(filter(None, [_xml_text(author.find("first")), _xml_text(author.find("last"))]))
                for author in entry.findall("author")
            ],
            abstract=_xml_text(entry.find("abstract")),
            url=f"https://aclanthology.org/{identifier}/",
        ))
    return papers


def _cvpr_papers() -> list[Paper]:
    catalogue = html.fromstring(_get(CVPR_PROCEEDINGS))
    papers = []
    for link in catalogue.xpath("//dt[@class='ptitle']/a[starts-with(@href, '/content/CVPR2026/html/')]"):
        papers.append(Paper(
            source="CVPR",
            title=link.text_content().strip(),
            authors=link.xpath("../following-sibling::dd[1]//input[@name='query_author']/@value"),
            abstract="",
            url=urljoin(CVPR_PROCEEDINGS, link.attrib["href"]),
        ))
    return papers


def _icml_papers() -> list[Paper]:
    catalogue = html.fromstring(_get(ICML_PROCEEDINGS))
    papers = []
    for entry in catalogue.xpath("//div[@class='paper'][p[@class='title']]"):
        links = entry.xpath(".//a[normalize-space()='abs']/@href")
        if links and links[0].startswith(ICML_PROCEEDINGS):
            authors = entry.xpath("string(.//span[@class='authors'])")
            papers.append(Paper(
                source="ICML",
                title=entry.xpath("string(p[@class='title'])").strip(),
                authors=[author.strip() for author in authors.split(",") if author.strip()],
                abstract="",
                url=links[0],
            ))
    return papers


def _virtual_papers(venue: str) -> list[Paper]:
    host = f"https://{venue.lower()}.cc"
    url = f"{host}/static/virtual/data/{venue.lower()}-2026-orals-posters.json"
    main_track = f"https://openreview.net/group?id={venue}.cc/2026/Conference"
    papers = []
    for entry in json.loads(_get(url))["results"]:
        path = entry.get("virtualsite_url") or ""
        decision = entry.get("decision") or ""
        if (
            entry.get("sourceurl") != main_track
            or not entry.get("visible")
            or not decision.casefold().startswith("accept (")
            or not path.startswith("/virtual/2026/poster/")
        ):
            continue
        papers.append(Paper(
            source=venue,
            title=entry["name"],
            authors=[author["fullname"] for author in entry.get("authors") or [] if author.get("fullname")],
            abstract="",
            url=urljoin(host, path),
        ))
    return papers


def _aaai_papers() -> list[Paper]:
    papers = []
    page_size = 1000
    # The 2026 volume has fewer than 10,000 records, within Crossref's offset limit.
    for offset in range(0, 10000, page_size):
        query = urlencode({
            "filter": "issn:2374-3468,from-pub-date:2026-01-01,until-pub-date:2026-12-31",
            "select": "DOI,title,author,abstract,volume,issue",
            "rows": page_size,
            "offset": offset,
            "sort": "published",
            "order": "asc",
        })
        response = json.loads(_get(f"https://api.crossref.org/works?{query}"))["message"]
        for entry in response["items"]:
            # Issues 1–43 are technical tracks, 44–45 are full-paper special tracks.
            # From issue 46 onward AAAI mixes those papers with companion material.
            if entry.get("volume") != "40" or entry.get("issue") not in AAAI_FULL_PAPER_ISSUES:
                continue
            titles = entry.get("title")
            doi = entry.get("DOI", "")
            if not titles or not doi.startswith("10.1609/aaai.v40i"):
                continue
            abstract = entry.get("abstract")
            papers.append(Paper(
                source="AAAI",
                title=titles[0],
                authors=[
                    " ".join(filter(None, [author.get("given"), author.get("family")]))
                    for author in entry.get("author", [])
                ],
                abstract=html.fromstring(abstract).text_content().strip() if abstract else "",
                url=f"https://doi.org/{doi}",
            ))
        if len(response["items"]) < page_size:
            break
    return papers


def fetch_conference_papers() -> list[Paper]:
    """Fetch public conference metadata, retaining the sources that respond."""
    sources = {
        "ACL": _acl_papers,
        "CVPR": _cvpr_papers,
        "ICML": _icml_papers,
        "ICLR": lambda: _virtual_papers("ICLR"),
        "NeurIPS": lambda: _virtual_papers("NeurIPS"),
        "AAAI": _aaai_papers,
    }
    papers = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending = {venue: pool.submit(fetch) for venue, fetch in sources.items()}
        for venue, future in pending.items():
            try:
                catalogue = future.result()
            except (OSError, ValueError, KeyError, ElementTree.ParseError, etree.LxmlError) as error:
                logger.warning(f"Unable to load {venue} 2026 conference papers: {error}")
                continue
            logger.info(f"Loaded {len(catalogue)} {venue} 2026 conference papers")
            papers.extend(catalogue)
    return papers


def daily_conference_candidates(
    catalogue: list[Paper], excluded_titles: list[str], day: date,
) -> list[Paper]:
    """Assign each title to a stable date bucket independently of catalogue order."""
    excluded = {_title_key(title) for title in excluded_titles}
    candidates: dict[int, Paper] = {}
    bucket = day.toordinal() % ROTATION_DAYS
    for paper in catalogue:
        key = _title_key(paper.title)
        if not key or key in excluded:
            continue
        fingerprint = int.from_bytes(sha256(key.encode()).digest())
        if fingerprint % ROTATION_DAYS == bucket:
            previous = candidates.get(fingerprint)
            if previous is None or (not previous.abstract and paper.abstract):
                candidates[fingerprint] = paper
    return [candidates[key] for key in sorted(candidates)[:CANDIDATE_LIMIT]]
