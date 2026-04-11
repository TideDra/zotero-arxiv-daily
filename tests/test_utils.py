"""Tests for zotero_arxiv_daily.utils: glob_match, send_email, tex extraction."""

import smtplib
import tarfile
import io

import pytest

from zotero_arxiv_daily.utils import glob_match, send_email, extract_tex_code_from_tar
from tests.canned_responses import make_stub_smtp


# ---------------------------------------------------------------------------
# glob_match — migrated from test_glob_match.py
# ---------------------------------------------------------------------------


class TestGlobMatch:
    """Test cases for the glob_match function."""

    def test_exact_match(self):
        assert glob_match("hello.txt", "hello.txt")
        assert not glob_match("hello.txt", "world.txt")
        assert glob_match("", "")

    def test_wildcard_asterisk(self):
        assert glob_match("hello.txt", "*.txt")
        assert not glob_match("hello.py", "*.txt")
        assert glob_match("file", "*")

        assert glob_match("hello.world.txt", "*.*.txt")
        assert not glob_match("hello.txt", "*.*.txt")
        assert glob_match("a.b.c.d", "*.*.*.*")

        assert glob_match("hello_world.txt", "hello*world.txt")
        assert glob_match("hello123world.txt", "hello*world.txt")
        assert glob_match("helloworld.txt", "hello*world.txt")
        assert not glob_match("hello_universe.txt", "hello*world.txt")

    def test_wildcard_question_mark(self):
        assert glob_match("hello.txt", "hell?.txt")
        assert not glob_match("hell.txt", "hell?.txt")
        assert glob_match("hello.txt", "he??o.txt")
        assert glob_match("heXXo.txt", "he??o.txt")
        assert not glob_match("heo.txt", "he??o.txt")

    def test_character_classes(self):
        assert glob_match("file1.txt", "file[123].txt")
        assert glob_match("file2.txt", "file[123].txt")
        assert not glob_match("file4.txt", "file[123].txt")

        assert glob_match("file1.txt", "file[1-3].txt")
        assert glob_match("file2.txt", "file[1-3].txt")
        assert not glob_match("file4.txt", "file[1-3].txt")

        assert glob_match("fileA.txt", "file[!123].txt")
        assert not glob_match("file1.txt", "file[!123].txt")

    def test_path_separators(self):
        assert glob_match("dir/file.txt", "dir/file.txt")
        assert glob_match("dir/file.txt", "*/file.txt")
        assert glob_match("dir/subdir/file.txt", "*/subdir/file.txt")
        assert glob_match("dir/subdir/file.txt", "dir/*/file.txt")
        assert glob_match("dir/subdir/file.txt", "*/*/file.txt")

    def test_complex_patterns(self):
        assert glob_match("test1_file.txt", "test[1-3]*file.txt")
        assert glob_match("test2_long_file.txt", "test[1-3]*file.txt")
        assert not glob_match("test4_file.txt", "test[1-3]*file.txt")

        assert glob_match("prefix_middle_suffix.log", "prefix*middle*.log")
        assert not glob_match("prefix_other_suffix.log", "prefix*middle*.log")

    def test_edge_cases(self):
        assert glob_match("", "**")
        assert not glob_match("", "?")
        assert not glob_match("file", "")

        assert glob_match("file-name.txt", "file-name.txt")
        assert glob_match("file_name.txt", "file_name.txt")
        assert glob_match("file.name.txt", "file.name.txt")

        assert not glob_match("File.txt", "file.txt")
        assert not glob_match("FILE.TXT", "file.txt")
        assert glob_match("file.txt", "file.txt")

    def test_partial_matches(self):
        assert glob_match("hello.txt", "hello.txt")
        assert not glob_match("prefix_hello.txt", "hello.txt")
        assert glob_match("hello.txt", "*.txt")

    def test_special_glob_characters(self):
        assert not glob_match("file[1].txt", "file[1].txt")
        assert glob_match("file1.txt", "file[1].txt")

    def test_numeric_patterns(self):
        assert glob_match("file001.txt", "file???.txt")
        assert not glob_match("file01.txt", "file???.txt")
        assert glob_match("version1.2.3.txt", "version*.txt")
        assert glob_match("version1.2.3.txt", "version?.?.?.txt")

    def test_extension_patterns(self):
        assert glob_match("document.pdf", "*.pdf")
        assert glob_match("image.jpg", "*.jpg")
        assert glob_match("script.py", "*.py")
        assert not glob_match("data.csv", "*.txt")

        assert glob_match("file.txt", "*.[tc][xs][tv]")
        assert glob_match("file.csv", "*.[tc][xs][tv]")
        assert not glob_match("file.pdf", "*.[tc][xs][tv]")

    def test_recursive_wildcard(self):
        assert glob_match("file.txt", "**/*.txt")
        assert glob_match("dir/file.txt", "**/*.txt")
        assert glob_match("dir/subdir/file.txt", "**/*.txt")
        assert glob_match("dir/subdir/subsubdir/file.txt", "**/*.txt")


# ---------------------------------------------------------------------------
# send_email
# ---------------------------------------------------------------------------


def test_send_email_starttls_success(config, monkeypatch):
    sent = []
    monkeypatch.setattr(smtplib, "SMTP", make_stub_smtp(sent))
    send_email(config, "<html>hello</html>")
    assert len(sent) == 1
    sender, recipients, body = sent[0]
    assert sender == "test@example.com"
    assert recipients == ["test@example.com"]
    # Body is a full MIME message (base64-encoded). Check the raw MIME string.
    assert "text/html" in body


def test_send_email_falls_back_to_ssl(config, monkeypatch):
    sent = []
    call_count = {"smtp": 0}

    StubOK = make_stub_smtp(sent)

    class StubSMTP_TLS_Fails:
        def __init__(self, *a, **kw):
            call_count["smtp"] += 1
        def starttls(self):
            raise OSError("TLS not supported")

    class StubSMTP_SSL(StubOK):
        pass

    monkeypatch.setattr(smtplib, "SMTP", StubSMTP_TLS_Fails)
    monkeypatch.setattr(smtplib, "SMTP_SSL", StubSMTP_SSL)
    send_email(config, "<html>ssl</html>")
    assert len(sent) == 1


def test_send_email_falls_back_to_plain(config, monkeypatch):
    sent = []
    call_count = {"smtp": 0}

    StubOK = make_stub_smtp(sent)

    class StubSMTP_TLS_Fails:
        def __init__(self, *a, **kw):
            call_count["smtp"] += 1
            if call_count["smtp"] == 1:
                pass  # first SMTP() call succeeds, but starttls will fail
            else:
                pass  # third SMTP() call is the plain fallback
        def starttls(self):
            raise OSError("TLS not supported")
        def login(self, u, p):
            pass
        def sendmail(self, s, r, m):
            sent.append((s, r, m))
        def quit(self):
            pass

    class StubSMTP_SSL_Fails:
        def __init__(self, *a, **kw):
            raise OSError("SSL not supported")

    monkeypatch.setattr(smtplib, "SMTP", StubSMTP_TLS_Fails)
    monkeypatch.setattr(smtplib, "SMTP_SSL", StubSMTP_SSL_Fails)
    send_email(config, "<html>plain</html>")
    assert len(sent) == 1


# ---------------------------------------------------------------------------
# extract_tex_code_from_tar
# ---------------------------------------------------------------------------


@pytest.fixture()
def make_tar(tmp_path):
    """Create a tar file with given files, auto-cleaned by tmp_path."""

    def _make(files: dict[str, str]) -> str:
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            for name, content in files.items():
                data = content.encode()
                info = tarfile.TarInfo(name=name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        path = tmp_path / "test.tar"
        path.write_bytes(buf.getvalue())
        return str(path)

    return _make


def test_extract_tex_single_file(make_tar):
    path = make_tar({"main.tex": "\\begin{document}\nHello\n\\end{document}"})
    result = extract_tex_code_from_tar(path, "test-paper")
    assert result is not None
    assert "Hello" in result["all"]


def test_extract_tex_with_input_resolution(make_tar):
    path = make_tar({
        "main.tex": "\\begin{document}\n\\input{intro}\n\\end{document}",
        "main.bbl": "",
        "intro.tex": "This is the introduction.",
    })
    result = extract_tex_code_from_tar(path, "test-paper")
    assert "This is the introduction." in result["all"]


def test_extract_tex_no_tex_files(make_tar):
    path = make_tar({"readme.md": "# Hello"})
    result = extract_tex_code_from_tar(path, "test-paper")
    assert result is None


def test_extract_tex_not_a_tar(tmp_path):
    path = tmp_path / "bad.tar"
    path.write_bytes(b"this is not a tar file")
    result = extract_tex_code_from_tar(str(path), "test-paper")
    assert result is None


def test_extract_tex_multiple_tex_no_bbl(make_tar):
    path = make_tar({
        "a.tex": "\\section{A}",
        "b.tex": "\\begin{document}\nMain content\n\\end{document}",
    })
    result = extract_tex_code_from_tar(path, "test-paper")
    assert result is not None
    assert "Main content" in result["all"]
