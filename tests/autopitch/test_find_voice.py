"""Tests for autopitch.scripts.find_voice (pure speaker-selection helpers)."""

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from autopitch.scripts.find_voice import (
    _hint_matches,
    _take_until_target,
    pick_dominant_speaker,
    pick_library_voice,
    score_library_voice,
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


class TestHintMatches:
    def test_gender_male_does_not_match_female(self):
        """Regression: 'male' is a substring of 'female'. Whole-word matching must
        not score the opposite gender as a match."""
        assert _hint_matches("male", "a warm female voice") is False
        assert _hint_matches("male", "female") is False

    def test_gender_male_matches_male(self):
        assert _hint_matches("male", "a deep male voice") is True
        assert _hint_matches("male", "male") is True

    def test_region_us_does_not_match_business(self):
        """'us' is a substring of 'business'/'focus' — must not match those."""
        assert _hint_matches("us", "our core business focus") is False
        assert _hint_matches("us", "us-based, american accent") is True

    def test_age_old_does_not_match_bold(self):
        assert _hint_matches("old", "a bold, golden tone") is False
        assert _hint_matches("old", "an old, gravelly voice") is True

    def test_case_insensitive(self):
        assert _hint_matches("MALE", "a Male narrator") is True

    def test_searches_multiple_fields(self):
        assert _hint_matches("british", None, "", "British narrator") is True
        assert _hint_matches("british", "american", "calm voice") is False

    def test_empty_or_none_hint(self):
        assert _hint_matches("", "anything") is False
        assert _hint_matches(None, "anything") is False

    def test_whitespace_hint_is_ignored(self):
        assert _hint_matches("  ", "anything") is False


class TestScoreLibraryVoice:
    def test_gender_match_via_description(self):
        female = {"name": "Bella", "description": "a young female voice", "category": "x"}
        assert score_library_voice(female, gender="female") == 3
        # The bug: scoring a female voice with gender='male' must NOT score the
        # gender component (old substring test matched because 'female' ⊃ 'male').
        assert score_library_voice(female, gender="male") == 0

    def test_gender_match_via_structured_label(self):
        """Many modern voices carry demographics only in `labels`, not the blurb."""
        v = {"name": "Adam", "description": "", "labels": {"gender": "male"}}
        assert score_library_voice(v, gender="male") == 3
        assert score_library_voice(v, gender="female") == 0

    def test_accent_label_matches_region(self):
        v = {"name": "Dorothy", "description": "", "labels": {"accent": "british"}}
        assert score_library_voice(v, region="british") == 2
        assert score_library_voice(v, region="american") == 0

    def test_age_label(self):
        v = {"name": "Arnold", "description": "", "labels": {"age": "old"}}
        assert score_library_voice(v, age="old") == 1
        assert score_library_voice(v, age="young") == 0

    def test_premade_category_tiebreak(self):
        plain = {"name": "X", "description": "", "category": "cloned"}
        premade = {"name": "Y", "description": "", "category": "premade"}
        generated = {"name": "Z", "description": "", "category": "generated"}
        assert score_library_voice(plain) == 0
        assert score_library_voice(premade) == 1
        assert score_library_voice(generated) == 1

    def test_combined_score(self):
        v = {
            "name": "Rachel",
            "description": "a calm young female voice",
            "labels": {"gender": "female", "accent": "american", "age": "young"},
            "category": "premade",
        }
        # gender(3) + region(2) + age(1) + premade(1)
        assert score_library_voice(v, gender="female", region="american", age="young") == 7

    def test_missing_fields_are_safe(self):
        assert score_library_voice({}, gender="male", region="us", age="old") == 0
        assert score_library_voice({"labels": None}, gender="male") == 0


class TestPickLibraryVoice:
    """Drives pick_library_voice with a fake ElevenLabs client so it never touches
    the network (and doesn't require the `elevenlabs` package to be installed)."""

    @staticmethod
    def _install_fake_client(monkeypatch, voices):
        mod = types.ModuleType("src.voice.elevenlabs_client")

        class FakeClient:
            def __init__(self, *a, **k):
                pass

            def list_voices(self):
                return list(voices)

        mod.ElevenLabsClient = FakeClient
        monkeypatch.setitem(sys.modules, "src.voice.elevenlabs_client", mod)

    def test_picks_correct_gender_end_to_end(self, monkeypatch):
        voices = [
            {"voice_id": "f1", "name": "Bella", "labels": {"gender": "female"}},
            {"voice_id": "m1", "name": "Adam", "labels": {"gender": "male"}},
        ]
        self._install_fake_client(monkeypatch, voices)
        # The whole point of the fix: asking for male returns the male voice, not
        # the female one (which the old substring scorer would have tied/picked).
        assert pick_library_voice(gender="male") == ("m1", "Adam")
        assert pick_library_voice(gender="female") == ("f1", "Bella")

    def test_returns_none_when_no_voices(self, monkeypatch):
        self._install_fake_client(monkeypatch, [])
        assert pick_library_voice(gender="male") is None

    def test_falls_back_to_first_on_no_match(self, monkeypatch):
        voices = [
            {"voice_id": "a", "name": "A", "category": "cloned"},
            {"voice_id": "b", "name": "B", "category": "cloned"},
        ]
        self._install_fake_client(monkeypatch, voices)
        # No hints match anything; max keeps the first voice in API order.
        assert pick_library_voice(gender="male") == ("a", "A")
