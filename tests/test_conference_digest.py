from datetime import date, timedelta
import json

from zotero_arxiv_daily import conference_digest

from tests.canned_responses import make_sample_paper


def test_daily_candidates_rotate_and_exclude_duplicate_titles():
    catalogue = [make_sample_paper(title=f"Conference Paper {i}", abstract="") for i in range(4000)]
    day = date(2026, 11, 15)
    today = conference_digest.daily_conference_candidates(catalogue, [], day)
    tomorrow = conference_digest.daily_conference_candidates(catalogue, [], day + timedelta(days=1))
    assert len(today) >= 20 and len(tomorrow) >= 20
    assert {paper.title for paper in today}.isdisjoint(paper.title for paper in tomorrow)

    saved = today[0].title.upper() + "."
    preprint = today[1].title
    repeat = make_sample_paper(title=today[2].title + "!", abstract="Published abstract")
    updated = conference_digest.daily_conference_candidates(
        [repeat, *reversed(catalogue)], [saved, preprint], day,
    )
    assert updated == [repeat, *today[3:]]


def test_virtual_catalogue_requires_visible_accepted_main_track_papers(monkeypatch):
    main = {
        "name": "An accepted paper",
        "sourceurl": "https://openreview.net/group?id=NeurIPS.cc/2026/Conference",
        "visible": True,
        "decision": "Accept (poster)",
        "virtualsite_url": "/virtual/2026/poster/123",
        "authors": [{"fullname": "A. Researcher"}],
    }
    entries = [
        main,
        main | {"name": "Rejected", "decision": "Reject"},
        main | {"name": "Withdrawn", "visible": False},
        main | {"name": "Workshop", "sourceurl": main["sourceurl"] + "/Workshop"},
        main | {"name": "Oral duplicate", "virtualsite_url": "/virtual/2026/oral/123"},
    ]
    monkeypatch.setattr(conference_digest, "_get", lambda _: json.dumps({"results": entries}).encode())

    paper, = conference_digest._virtual_papers("NeurIPS")
    assert paper.title == main["name"]
    assert paper.authors == ["A. Researcher"]
    assert paper.url == "https://neurips.cc/virtual/2026/poster/123"


def test_one_unavailable_venue_does_not_stop_the_others(monkeypatch):
    def unavailable():
        raise TimeoutError("catalogue timed out")

    acl = make_sample_paper(source="ACL")
    monkeypatch.setattr(conference_digest, "_acl_papers", lambda: [acl])
    monkeypatch.setattr(conference_digest, "_cvpr_papers", unavailable)
    monkeypatch.setattr(conference_digest, "_icml_papers", lambda: [])
    monkeypatch.setattr(conference_digest, "_virtual_papers", lambda _: [])
    monkeypatch.setattr(conference_digest, "_aaai_papers", lambda: [])

    assert conference_digest.fetch_conference_papers() == [acl]
