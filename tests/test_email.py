from email import message_from_string

import pytest

from zotero_arxiv_daily.construct_email import render_email
from zotero_arxiv_daily.protocol import Paper
from zotero_arxiv_daily.topic_clusterer import PaperGroup
from zotero_arxiv_daily.utils import send_email


@pytest.fixture
def papers() -> list[Paper]:
    return [
        Paper(
            source="arxiv",
            title="Vision Paper One",
            authors=["Author A", "Author B"],
            abstract="Test Abstract 1",
            url="https://arxiv.org/abs/2512.00001",
            pdf_url="https://arxiv.org/pdf/2512.00001",
            full_text="Test Full Text 1",
            tldr="Test TLDR 1",
            affiliations=["Affiliation 1", "Affiliation 2"],
            score=9.4,
        ),
        Paper(
            source="arxiv",
            title="Vision Paper Two",
            authors=["Author C", "Author D"],
            abstract="Test Abstract 2",
            url="https://arxiv.org/abs/2512.00002",
            pdf_url="https://arxiv.org/pdf/2512.00002",
            full_text="Test Full Text 2",
            tldr="Test TLDR 2",
            affiliations=["Affiliation 3", "Affiliation 4"],
            score=8.7,
        ),
        Paper(
            source="arxiv",
            title="Agent Paper One",
            authors=["Author E", "Author F"],
            abstract="Test Abstract 3",
            url="https://arxiv.org/abs/2512.00003",
            pdf_url="https://arxiv.org/pdf/2512.00003",
            full_text="Test Full Text 3",
            tldr="Test TLDR 3",
            affiliations=["Affiliation 5", "Affiliation 6"],
            score=8.1,
        ),
        Paper(
            source="arxiv",
            title="Agent Paper Two",
            authors=["Author G", "Author H"],
            abstract="Test Abstract 4",
            url="https://arxiv.org/abs/2512.00004",
            pdf_url="https://arxiv.org/pdf/2512.00004",
            full_text="Test Full Text 4",
            tldr="Test TLDR 4",
            affiliations=["Affiliation 7", "Affiliation 8"],
            score=7.8,
        ),
    ]


class FakeSMTP:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.logged_in = None
        self.sent = None
        self.closed = False

    def starttls(self):
        return None

    def login(self, sender: str, password: str):
        self.logged_in = (sender, password)

    def sendmail(self, sender: str, receivers: list[str], message: str):
        self.sent = (sender, receivers, message)

    def quit(self):
        self.closed = True


class UnexpectedSMTPSSL:
    def __init__(self, *args, **kwargs):
        raise AssertionError("SMTP_SSL should not be used in this unit test")


def test_render_email_renders_group_headings_and_summaries(papers: list[Paper]):
    groups = [
        PaperGroup(label="Vision", summary="Papers about visual understanding.", papers=papers[:2]),
        PaperGroup(label="Agents", summary="Papers about agents.", papers=papers[2:4]),
    ]

    email_content = render_email(groups)

    assert "Vision" in email_content
    assert "Papers about visual understanding." in email_content
    assert "Agents" in email_content
    assert "Papers about agents." in email_content
    assert papers[0].title in email_content
    assert papers[2].title in email_content
    assert email_content.index(papers[0].title) < email_content.index(papers[1].title)
    assert email_content.index(papers[2].title) < email_content.index(papers[3].title)


def test_render_email_renders_fallback_group_without_summary(papers: list[Paper]):
    groups = [PaperGroup(label="Relevant papers today", summary=None, papers=[papers[1], papers[0]])]

    email_content = render_email(groups)

    assert "Relevant papers today" in email_content
    assert "<strong>TLDR:</strong>" in email_content
    assert papers[1].title in email_content
    assert papers[0].title in email_content
    assert email_content.index(papers[1].title) < email_content.index(papers[0].title)
    assert "None" not in email_content


def test_render_email_escapes_group_label_and_summary_html(papers: list[Paper]):
    groups = [
        PaperGroup(
            label="Vision <Agents>",
            summary='<script>alert("x")</script> & more',
            papers=[papers[0]],
        )
    ]

    email_content = render_email(groups)

    assert "Vision &lt;Agents&gt;" in email_content
    assert "&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt; &amp; more" in email_content
    assert "Vision <Agents>" not in email_content
    assert '<script>alert("x")</script> & more' not in email_content


def test_render_email_escapes_grouped_paper_fields_and_pdf_url(papers: list[Paper]):
    dangerous_paper = Paper(
        source="arxiv",
        title="Paper <One>",
        authors=['Author <A>', 'Author "B"'],
        abstract="Abstract",
        url="https://example.com/abs",
        pdf_url='https://example.com/pdf?paper=1&redirect="bad"',
        full_text="Full text",
        tldr='TLDR <script>alert("x")</script> & more',
        affiliations=['Affiliation <One>', 'Affiliation & Two'],
        score=9.1,
    )
    groups = [PaperGroup(label="Vision", summary="Summary", papers=[dangerous_paper])]

    email_content = render_email(groups)

    assert "Paper &lt;One&gt;" in email_content
    assert "Author &lt;A&gt;, Author &quot;B&quot;" in email_content
    assert "TLDR &lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt; &amp; more" in email_content
    assert "Affiliation &lt;One&gt;, Affiliation &amp; Two" in email_content
    assert 'href="https://example.com/pdf?paper=1&amp;redirect=&quot;bad&quot;"' in email_content
    assert "Paper <One>" not in email_content
    assert 'TLDR <script>alert("x")</script> & more' not in email_content
    assert 'href="https://example.com/pdf?paper=1&redirect="bad""' not in email_content




def test_render_email_preserves_empty_email_behavior():
    email_content = render_email([])

    assert "No Papers Today. Take a Rest!" in email_content


def test_send_email(config, papers: list[Paper], monkeypatch: pytest.MonkeyPatch):
    groups = [PaperGroup(label="Vision", summary="Papers about visual understanding.", papers=papers[:2])]
    html = render_email(groups)
    fake_server = FakeSMTP(config.email.smtp_server, config.email.smtp_port)

    monkeypatch.setattr("zotero_arxiv_daily.utils.smtplib.SMTP", lambda host, port: fake_server)
    monkeypatch.setattr("zotero_arxiv_daily.utils.smtplib.SMTP_SSL", UnexpectedSMTPSSL)

    assert "Vision" in html

    send_email(config, html)

    assert fake_server.logged_in == (config.email.sender, config.email.sender_password)
    assert fake_server.closed is True
    sender, receivers, raw_message = fake_server.sent
    assert sender == config.email.sender
    assert receivers == [config.email.receiver]

    parsed = message_from_string(raw_message)
    assert parsed["To"] is not None
    assert parsed["Subject"] is not None
    assert parsed.get_content_type() == "text/html"
    decoded_html = parsed.get_payload(decode=True).decode("utf-8")
    assert "Vision" in decoded_html
    assert papers[0].title in decoded_html
