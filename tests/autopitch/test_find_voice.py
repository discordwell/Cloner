"""Tests for autopitch.scripts.find_voice (pure speaker-selection helpers)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from autopitch.scripts.find_voice import (
    _take_until_target,
    pick_dominant_speaker,
    segments_for_speaker,
)


class TestPickDominantSpeaker:
    def test_single_speaker(self):
        segs = [("A", 0.0, 30.0)]
        assert pick_dominant_speaker(segs) == "A"

    def test_interview_format(self):
        """Classic solo interview: interviewee speaks ~80% of the time."""
        segs = [
            ("interviewer", 0.0, 5.0),   # 5s question
            ("guest", 5.0, 65.0),         # 60s answer
            ("interviewer", 65.0, 70.0), # 5s question
            ("guest", 70.0, 120.0),       # 50s answer
        ]
        assert pick_dominant_speaker(segs) == "guest"

    def test_balanced_converstaion_picks_longer(self):
        segs = [
            ("A", 0.0, 40.0),
            ("B", 40.0, 75.0),
        ]
        assert pick_dominant_speaker(segs) == "A"


class TestSegmentsForSpeaker:
    def test_takes_segments_until_target_reached(self):
        segs = [
            ("guest", 0.0, 20.0),
            ("host", 20.0, 25.0),
            ("guest", 25.0, 60.0),
            ("host", 60.0, 70.0),
            ("guest", 70.0, 120.0),
        ]
        picked = segments_for_speaker(segs, "guest", target_s=40.0)
        # First two guest segments total 20 + 35 = 55s > 40s target
        # So we get segment 1 (20s) + a trim of segment 2 to 20s more.
        total = sum(e - s for s, e in picked)
        assert abs(total - 40.0) < 0.01
        assert picked[0] == (0.0, 20.0)
        assert picked[1][0] == 25.0  # second guest segment start

    def test_sorted_by_start(self):
        segs = [
            ("guest", 50.0, 80.0),
            ("guest", 0.0, 20.0),
        ]
        picked = segments_for_speaker(segs, "guest", target_s=100)
        assert picked[0][0] < picked[1][0]

    def test_skips_other_speakers(self):
        segs = [
            ("a", 0.0, 30.0),
            ("b", 30.0, 60.0),
        ]
        picked = segments_for_speaker(segs, "b", target_s=100)
        assert picked == [(30.0, 60.0)]

    def test_exact_fill_emits_no_zero_length_segment(self):
        """Regression: when a speaker's segments fill target_s exactly, the next
        segment used to be appended as a zero-length (s, s) span. That becomes a
        degenerate `atrim=start=s:end=s` (empty input) in extract_segments'
        ffmpeg concat filtergraph."""
        segs = [("A", 0.0, 10.0), ("A", 12.0, 17.0)]
        picked = segments_for_speaker(segs, "A", target_s=10.0)
        assert picked == [(0.0, 10.0)]
        assert all(e - s > 0 for s, e in picked)


class TestTakeUntilTarget:
    def test_trims_final_span_to_target(self):
        out = _take_until_target([(0.0, 8.0), (8.0, 20.0)], target_s=10.0)
        assert out == [(0.0, 8.0), (8.0, 10.0)]
        assert sum(e - s for s, e in out) == 10.0

    def test_stops_on_exact_boundary_without_empty_span(self):
        out = _take_until_target([(0.0, 5.0), (5.0, 10.0), (10.0, 15.0)], target_s=10.0)
        assert out == [(0.0, 5.0), (5.0, 10.0)]
        assert all(e - s > 0 for s, e in out)

    def test_skips_degenerate_input_spans(self):
        out = _take_until_target([(3.0, 3.0), (3.0, 6.0)], target_s=100.0)
        assert out == [(3.0, 6.0)]

    def test_undershoot_returns_all_spans(self):
        out = _take_until_target([(0.0, 2.0), (5.0, 6.0)], target_s=100.0)
        assert out == [(0.0, 2.0), (5.0, 6.0)]

    def test_empty_input(self):
        assert _take_until_target([], target_s=10.0) == []
