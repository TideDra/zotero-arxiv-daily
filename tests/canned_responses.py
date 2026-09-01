"""Shared stub factories for tests. No unittest.mock anywhere."""

from datetime import datetime
from types import SimpleNamespace

from zotero_arxiv_daily.protocol import CorpusPaper, Paper


# ---------------------------------------------------------------------------
# OpenAI client stub
# ---------------------------------------------------------------------------

_AFFILIATION_MARKER = "You are an assistant who perfectly extracts affiliations"
_AFFILIATION_RESPONSE = '["TsingHua University","Peking University"]'
_TLDR_RESPONSE = "Hello! How can I assist you today?"


def _make_chat_response(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason="stop",
                index=0,
            )
        ],
        id="chatcmpl-stub",
        created=1765197615,
        model="gpt-4o-mini-2024-07-18",
        object="chat.completion",
    )


def _stub_chat_create(**kwargs):
    messages = kwargs.get("messages", [])
    request_str = str(messages)
    if _AFFILIATION_MARKER in request_str:
        return _make_chat_response(_AFFILIATION_RESPONSE)
    return _make_chat_response(_TLDR_RESPONSE)


def _stub_response_create(**kwargs):
    request_str = str(kwargs.get("input", []))
    content = _AFFILIATION_RESPONSE if _AFFILIATION_MARKER in request_str else _TLDR_RESPONSE
    return SimpleNamespace(output_text=content)


def _stub_embeddings_create(**kwargs):
    inputs = kwargs.get("input", [])
    n = len(inputs) if isinstance(inputs, list) else 1
    return SimpleNamespace(
        data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3], index=i, object="embedding") for i in range(n)],
        model="text-embedding-3-large",
        object="list",
    )


def make_stub_openai_client():
    """Return a SimpleNamespace that quacks like openai.OpenAI().

    chat.completions.create(), responses.create(), and embeddings.create() behave identically
    to the Docker mock_openai server that CI previously relied on.
    """
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=_stub_chat_create),
        ),
        responses=SimpleNamespace(create=_stub_response_create),
        embeddings=SimpleNamespace(create=_stub_embeddings_create),
    )


# ---------------------------------------------------------------------------
# Zotero client stub
# ---------------------------------------------------------------------------

_DEFAULT_COLLECTIONS = [
    {
        "key": "COL1",
        "data": {"name": "survey", "parentCollection": False},
    },
    {
        "key": "COL2",
        "data": {"name": "topic-a", "parentCollection": "COL1"},
    },
]

_DEFAULT_ITEMS = [
    {
        "data": {
            "title": "Stub Paper 1",
            "abstractNote": "Abstract of stub paper 1.",
            "dateAdded": "2026-01-15T10:00:00Z",
            "collections": ["COL2"],
        },
    },
    {
        "data": {
            "title": "Stub Paper 2",
            "abstractNote": "Abstract of stub paper 2.",
            "dateAdded": "2026-02-20T12:00:00Z",
            "collections": ["COL1"],
        },
    },
]


def make_stub_zotero_client(collections=None, items=None):
    """Return a SimpleNamespace that quacks like pyzotero.zotero.Zotero.

    Supports the call patterns used by Executor.fetch_zotero_corpus():
        zot.everything(zot.collections())
        zot.everything(zot.items(itemType=...))
    """
    cols = collections if collections is not None else _DEFAULT_COLLECTIONS
    itms = items if items is not None else _DEFAULT_ITEMS

    def everything(generator):
        return generator

    def collections_fn():
        return cols

    def items_fn(**kwargs):
        return itms

    return SimpleNamespace(
        everything=everything,
        collections=collections_fn,
        items=items_fn,
    )


# ---------------------------------------------------------------------------
# SMTP stub
# ---------------------------------------------------------------------------


def make_stub_smtp(sent_emails: list):
    """Return a class that records calls to sendmail().

    Usage:
        sent = []
        monkeypatch.setattr(smtplib, "SMTP", make_stub_smtp(sent))
        ...
        assert len(sent) == 1
        sender, recipients, body = sent[0]
    """

    class StubSMTP:
        def __init__(self, *args, **kwargs):
            pass

        def starttls(self):
            pass

        def login(self, user, password):
            pass

        def sendmail(self, sender, recipients, msg):
            sent_emails.append((sender, recipients, msg))

        def quit(self):
            pass

    return StubSMTP


# ---------------------------------------------------------------------------
# Paper / CorpusPaper factories
# ---------------------------------------------------------------------------


def make_sample_paper(**overrides) -> Paper:
    defaults = dict(
        source="arxiv",
        title="Sample Paper Title",
        authors=["Author A", "Author B", "Author C"],
        abstract="This paper explores a novel approach to widget engineering.",
        url="https://arxiv.org/abs/2026.00001",
        pdf_url="https://arxiv.org/pdf/2026.00001",
        full_text="\\begin{document} Some text. \\end{document}",
        tldr=None,
        affiliations=None,
        score=None,
    )
    defaults.update(overrides)
    return Paper(**defaults)


def make_sample_corpus(n: int = 3) -> list[CorpusPaper]:
    return [
        CorpusPaper(
            title=f"Corpus Paper {i}",
            abstract=f"Abstract for corpus paper {i}.",
            added_date=datetime(2026, 1, 1 + i),
            paths=[f"2026/survey/topic-{i}"],
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# bioRxiv canned API response
# ---------------------------------------------------------------------------

SAMPLE_BIORXIV_API_RESPONSE = {
    "messages": [{"status": "ok"}],
    "collection": [
        {
            "doi": "10.1101/2026.03.01.000001",
            "title": "A biorxiv paper",
            "authors": "Smith, J.; Doe, A.; Lee, K.",
            "abstract": "We present a novel finding.",
            "date": "2026-03-02",
            "category": "bioinformatics",
            "version": "1",
        },
        {
            "doi": "10.1101/2026.03.01.000002",
            "title": "Another biorxiv paper",
            "authors": "Wang, L.; Chen, M.",
            "abstract": "We replicate a key result.",
            "date": "2026-03-02",
            "category": "genomics",
            "version": "1",
        },
        {
            "doi": "10.1101/2026.03.01.000003",
            "title": "Old biorxiv paper",
            "authors": "Old, R.",
            "abstract": "Yesterday's paper.",
            "date": "2026-03-01",
            "category": "bioinformatics",
            "version": "1",
        },
    ],
}


# ---------------------------------------------------------------------------
# chemRxiv canned API response (Crossref REST API, prefix 10.26434)
# ---------------------------------------------------------------------------

def _chemrxiv_item(doi, title, created, authors, abstract="<jats:p>An abstract.</jats:p>"):
    """Build a Crossref ``posted-content`` work record for a chemRxiv preprint.

    ``authors`` is a list of (given, family) tuples. ``created`` is an ISO timestamp.
    """
    return {
        "DOI": doi,
        "type": "posted-content",
        "subtype": "preprint",
        "publisher": "American Chemical Society (ACS)",
        "prefix": "10.26434",
        "title": [title],
        "abstract": abstract,
        "created": {"date-time": created, "timestamp": 0},
        "posted": {"date-parts": [[int(p) for p in created[:10].split("-")]]},
        "author": [
            {"given": given, "family": family, "sequence": "first" if i == 0 else "additional",
             "affiliation": [{"name": "Some University"}]}
            for i, (given, family) in enumerate(authors)
        ],
        "link": [{"URL": f"https://chemrxiv.org/doi/pdf/{doi}", "content-type": "unspecified"}],
        "resource": {"primary": {"URL": f"https://chemrxiv.org/doi/full/{doi}"}},
        "URL": f"https://doi.org/{doi}",
    }


def _chemrxiv_response(items, total=None):
    return {
        "status": "ok",
        "message-type": "work-list",
        "message": {
            "total-results": len(items) if total is None else total,
            "items": items,
            "items-per-page": 100,
            "query": {"start-index": 0},
        },
    }


SAMPLE_CHEMRXIV_API_RESPONSE = _chemrxiv_response([
    _chemrxiv_item(
        "10.26434/chemrxiv.15007618/v1",
        "A chemrxiv paper",
        "2026-03-02T10:00:00Z",
        [("Jane", "Smith"), ("Alan", "Doe")],
        abstract="<jats:p>We study RuO<jats:sub>2</jats:sub> &amp; friends.</jats:p>",
    ),
    _chemrxiv_item(
        "10.26434/chemrxiv.15007000/v2",
        "A revised chemrxiv paper",
        "2026-03-02T09:00:00Z",
        [("Li", "Wang")],
    ),
    _chemrxiv_item(
        "10.26434/chemrxiv.15007619/v1",
        "Another chemrxiv paper",
        "2026-03-02T08:00:00Z",
        [("Rip", "Old")],
    ),
    _chemrxiv_item(
        "10.26434/chemrxiv.15006000/v1",
        "Old chemrxiv paper",
        "2026-02-20T08:00:00Z",
        [("Rip", "Old")],
    ),
])
