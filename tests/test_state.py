from datetime import datetime, timezone
import json

import pytest

from tests.canned_responses import make_sample_paper
from zotero_arxiv_daily.state import AdsState


def test_ads_state_round_trip_and_filter(tmp_path):
    path = tmp_path / "state" / "ads.json"
    paper = make_sample_paper(
        source="ads",
        external_ids={"ads": "2026ApJ...999....1A"},
        entry_at=datetime(2026, 8, 22, tzinfo=timezone.utc),
    )
    state = AdsState()
    assert state.filter_new([paper]) == [paper]
    state.mark_seen([paper])
    state.save(path)

    loaded = AdsState.load(path)
    assert loaded.filter_new([paper]) == []
    payload = json.loads(path.read_text())
    assert set(payload) == {"version", "updated_at", "seen"}
    assert "abstract" not in path.read_text()


def test_legacy_ads_state_filters_scix_record(tmp_path):
    path = tmp_path / "state" / "ads.json"
    legacy_paper = make_sample_paper(
        source="ads",
        external_ids={"ads": "2026ApJ...999....1A"},
        entry_at=datetime(2026, 8, 22, tzinfo=timezone.utc),
    )
    state = AdsState()
    state.mark_seen([legacy_paper])
    state.save(path)

    scix_paper = make_sample_paper(
        source="scix",
        external_ids={"ads": "2026ApJ...999....1A"},
    )
    assert AdsState.load(path).filter_new([scix_paper]) == []


def test_ads_state_missing_is_empty(tmp_path):
    assert AdsState.load(tmp_path / "missing.json").seen == {}


def test_ads_state_rejects_corruption(tmp_path):
    path = tmp_path / "ads.json"
    path.write_text("not json")
    with pytest.raises(ValueError, match="Invalid SciX/ADS-compatible state"):
        AdsState.load(path)
