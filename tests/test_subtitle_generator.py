"""Subtitle generator: all 17 presets, ASS structure, karaoke + segment modes."""

import json

from processors import subtitle_generator as sg


def _transcript_words():
    words = []
    t = 0.5
    for text in ("This is the first caption line", "and here comes the second one",
                 "plus a third line to wrap around nicely"):
        for word in text.split():
            words.append({"word": word, "start": t, "end": t + 0.3})
            t += 0.4
    return words


def _hints(out_w=1080, out_h=1920):
    return {"geometry": {"out_w": out_w, "out_h": out_h, "mode": "crop",
                         "src_w": 640, "src_h": 360,
                         "target_label": "9:16", "z_max": 1.1,
                         "aspect_window": {"w": 202, "h": 360},
                         "manual_transform": None, "manual_window": None,
                         "output_quality": "original", "quality_note": "note"},
            "clips": {"1": {"zoom_windows": [], "crop_keypoints": None,
                            "caption_band": "bottom"}}}


def _clips():
    return [{"clip_index": 1, "start_sec": 0.0, "end_sec": 6.0, "viral_score": 80,
             "topic": "t", "hook": "", "title": "T", "description": "D",
             "tags": ["#a", "#b", "#c"], "visual_analysis": {}, "duration": 6.0,
             "editing": {}}]


def test_all_presets_present():
    expected = {"VIRAL", "PODCAST", "IMPACT", "NEON", "MAGENTA", "GOLD", "CRIMSON",
                "CYBER", "BEAST", "MARKER", "RETRO", "GLITCH", "MINIMAL", "EMERALD",
                "SUNSET", "VIOLET"}
    assert expected == set(sg.SUBTITLE_PRESETS.keys())
    # Fonts/colors/structure the Android contract requires.
    assert sg.SUBTITLE_PRESETS["VIRAL"]["font"] == "Montserrat"
    assert sg.SUBTITLE_PRESETS["VIRAL"]["highlight_color"] == "#FFD600"
    assert sg.SUBTITLE_PRESETS["IMPACT"]["font"] == "Anton"
    assert sg.SUBTITLE_PRESETS["CYBER"]["highlight_mode"] == "box"
    assert sg.SUBTITLE_PRESETS["MINIMAL"]["highlight_mode"] == "none"
    assert sg.SUBTITLE_PRESETS["BEAST"]["outline"] == 8.0


def test_none_style_disables_subtitles(tmp_path):
    subtitles = sg.generate_subtitles(
        {"subtitle_style": "NONE", "subtitle_language": ""}, _hints(), _clips(),
        {"language": "en", "segments": [], "words": []}, tmp_path)
    assert subtitles["style"] == "NONE"
    assert subtitles["clips"] == []


def test_karaoke_word_mode(tmp_path):
    transcript = {"language": "en", "word_timings_available": True,
                  "segments": [{"start": 0.5, "end": 3.0, "text": "x"}],
                  "words": _transcript_words()}
    params = {"subtitle_style": "VIRAL", "subtitle_language": ""}
    subtitles = sg.generate_subtitles(params, _hints(), _clips(), transcript, tmp_path)

    assert subtitles["word_timings_used"] is True
    assert subtitles["preset"]["font"] == "Montserrat"
    clip = subtitles["clips"][0]
    assert clip["burned_in"] is True
    assert clip["alignment"] == 2  # bottom band
    assert clip["cues"]

    ass = open(clip["file"], encoding="utf-8").read()
    assert "[Script Info]" in ass
    assert "PlayResX: 1080" in ass and "PlayResY: 1920" in ass
    assert "Style: Clipzy,Montserrat," in ass
    assert "karaoke" not in ass  # \k tags, not the word
    assert "\\k" in ass          # karaoke timing tags present
    assert "\\N" in ass          # wrapped lines
    assert ass.count("Dialogue: 0,") == len(clip["cues"])
    # VIRAL is caps: dialogue text (minus ASS override tags) must be upper-case.
    import re

    for line in ass.splitlines():
        if line.startswith("Dialogue:"):
            body = re.sub(r"\{[^}]*\}", "", line.split(",Clipzy,,", 1)[-1])
            letters = [ch for ch in body if ch.isalpha()]
            assert letters and all(ch.isupper() for ch in letters), body


def test_segment_mode_without_word_timings(tmp_path):
    transcript = {"language": "en", "word_timings_available": False,
                  "segments": [{"start": 0.5, "end": 3.0,
                                "text": "A segment without word timings to chunk"}],
                  "words": []}
    params = {"subtitle_style": "MINIMAL", "subtitle_language": ""}
    subtitles = sg.generate_subtitles(params, _hints(), _clips(), transcript, tmp_path)
    assert subtitles["word_timings_used"] is False
    assert subtitles["word_source"] == "whisper_segments"
    ass = open(subtitles["clips"][0]["file"], encoding="utf-8").read()
    assert "\\k" not in ass  # no fabricated word synchronization
    assert "Style: Clipzy,Inter," in ass


def test_caption_band_top_changes_alignment(tmp_path):
    transcript = {"language": "en", "word_timings_available": True,
                  "segments": [], "words": _transcript_words()}
    hints = _hints()
    hints["clips"]["1"]["caption_band"] = "top"
    subtitles = sg.generate_subtitles(
        {"subtitle_style": "PODCAST", "subtitle_language": ""}, hints, _clips(),
        transcript, tmp_path)
    assert subtitles["clips"][0]["alignment"] == 8


def test_ass_time_format():
    assert sg.fmt_ass_time(0) == "0:00:00.00"
    assert sg.fmt_ass_time(61.25) == "0:01:01.25"
    assert sg.fmt_ass_time(3661.5) == "1:01:01.50"
    assert sg.fmt_ass_time(-5) == "0:00:00.00"


def test_hex_to_ass():
    assert sg.hex_to_ass("#FFFFFF") == "&H00FFFFFF"
    assert sg.hex_to_ass("#FFD600") == "&H0000D6FF"  # BGR order
    assert sg.hex_to_ass("#F00", alpha="70") == "&H700000FF"
