from zotero_arxiv_daily.retriever.arxiv_retriever import ArxivRetriever
from zotero_arxiv_daily.retriever.arxiv_retriever import RSSPaper
from zotero_arxiv_daily.retriever.arxiv_retriever import download_url_to_file
from zotero_arxiv_daily.retriever.arxiv_retriever import extract_text_from_pdf_with_timeout
import feedparser

def test_arxiv_retriever(config, monkeypatch):

    parsed_result = feedparser.parse("tests/retriever/arxiv_rss_example.xml")
    raw_parser = feedparser.parse
    def mock_feedparser_parse(url):
        if url == f"https://rss.arxiv.org/atom/{'+'.join(config.source.arxiv.category)}":
            return parsed_result
        return raw_parser(url)
    monkeypatch.setattr(feedparser, "parse", mock_feedparser_parse)
    
    retriever = ArxivRetriever(config)
    papers = retriever.retrieve_papers()
    parsed_results = [i for i in parsed_result.entries if i.get("arxiv_announce_type","new") == 'new']
    assert len(papers) == len(parsed_results)
    paper_titles = [i.title for i in papers]
    parsed_titles = [i.title for i in parsed_results]
    assert set(paper_titles) == set(parsed_titles)

def test_extract_text_from_pdf_timeout_returns_none(monkeypatch):
    paper = RSSPaper(
        entry_id="1234.5678",
        title="Stuck PDF",
        summary="summary",
        authors=["A"],
        pdf_url="https://example.com/paper.pdf",
    )

    class FakeQueue:
        def get_nowait(self):
            raise Exception("queue should be empty on timeout")

    class FakeProcess:
        def __init__(self, target, args):
            self.target = target
            self.args = args
            self.terminated = False

        def start(self):
            pass

        def join(self, timeout=None):
            pass

        def is_alive(self):
            return True

        def terminate(self):
            self.terminated = True

    class FakeContext:
        def __init__(self):
            self.process = None

        def Queue(self, maxsize=1):
            return FakeQueue()

        def Process(self, target, args):
            self.process = FakeProcess(target, args)
            return self.process

    fake_context = FakeContext()
    monkeypatch.setattr("zotero_arxiv_daily.retriever.arxiv_retriever.get_context", lambda method: fake_context)

    assert extract_text_from_pdf_with_timeout(paper) is None
    assert fake_context.process is not None
    assert fake_context.process.terminated is True

def test_extract_text_from_pdf_timeout_uses_fork_on_posix(monkeypatch):
    paper = RSSPaper(
        entry_id="1234.5678",
        title="Stuck PDF",
        summary="summary",
        authors=["A"],
        pdf_url="https://example.com/paper.pdf",
    )
    called = {}

    class FakeQueue:
        def get_nowait(self):
            raise Exception("queue should be empty on timeout")

    class FakeProcess:
        def start(self):
            pass

        def join(self, timeout=None):
            pass

        def is_alive(self):
            return True

        def terminate(self):
            pass

    class FakeContext:
        def Queue(self, maxsize=1):
            return FakeQueue()

        def Process(self, target, args):
            return FakeProcess()

    def fake_get_context(method):
        called["method"] = method
        return FakeContext()

    monkeypatch.setattr("zotero_arxiv_daily.retriever.arxiv_retriever.get_context", fake_get_context)
    monkeypatch.setattr("zotero_arxiv_daily.retriever.arxiv_retriever.os.name", "posix")

    assert extract_text_from_pdf_with_timeout(paper) is None
    assert called["method"] == "fork"

def test_download_url_to_file_passes_timeout(monkeypatch, tmp_path):
    called = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self, size=-1):
            return b""

    def fake_urlopen(url, timeout):
        called["url"] = url
        called["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("zotero_arxiv_daily.retriever.arxiv_retriever.urlopen", fake_urlopen)

    output = tmp_path / "paper.pdf"
    download_url_to_file("https://example.com/paper.pdf", str(output), 12)

    assert called == {"url": "https://example.com/paper.pdf", "timeout": 12}
