"""Styled ASS subtitle generation (port of PY_ASS with all 17 presets).

Android SubtitleStyles.kt preset replicas (identifiers are the contract). If
SubtitleStyles.kt changes, sync ONLY the SUBTITLE_PRESETS table: keys map 1:1
to the Android preset fields (font, weight, italic, colors, outline, shadow,
caps, highlight mode, line spacing, letter spacing, relative size).
line_spacing is approximated by the renderer (ASS has no direct line-spacing
control) but kept here for fidelity with the Android definition.
"""

from __future__ import annotations

from utils.logging_setup import get_logger

logger = get_logger("subtitles")

SUBTITLE_PRESETS = {
    "VIRAL": {
        "font": "Montserrat", "weight": 800, "italic": False, "char_width": 0.58,
        "text_color": "#FFFFFF", "highlight_color": "#FFD600",
        "outline_color": "#000000", "outline": 5.0,
        "shadow_color": "#000000", "shadow_radius": 0.0,
        "caps": True, "highlight_mode": "karaoke_word",
        "line_spacing": 1.05, "letter_spacing": 0.5,
        "size_ratio": 0.055, "margin_ratio": 0.12,
    },
    "PODCAST": {
        "font": "Roboto", "weight": 500, "italic": False, "char_width": 0.52,
        "text_color": "#FFFFFF", "highlight_color": "#4ADE80",
        "outline_color": "#101010", "outline": 3.0,
        "shadow_color": "#000000", "shadow_radius": 2.0,
        "caps": False, "highlight_mode": "karaoke_word",
        "line_spacing": 1.15, "letter_spacing": 0.0,
        "size_ratio": 0.046, "margin_ratio": 0.10,
    },
    "IMPACT": {
        "font": "Anton", "weight": 400, "italic": False, "char_width": 0.50,
        "text_color": "#FFFFFF", "highlight_color": "#FF3B30",
        "outline_color": "#000000", "outline": 6.0,
        "shadow_color": "#000000", "shadow_radius": 0.0,
        "caps": True, "highlight_mode": "karaoke_word",
        "line_spacing": 1.0, "letter_spacing": 1.0,
        "size_ratio": 0.062, "margin_ratio": 0.13,
    },
    "NEON": {
        "font": "Orbitron", "weight": 600, "italic": False, "char_width": 0.62,
        "text_color": "#00E5FF", "highlight_color": "#FF00E5",
        "outline_color": "#00B8D4", "outline": 3.0,
        "shadow_color": "#00E5FF", "shadow_radius": 6.0,
        "caps": False, "highlight_mode": "karaoke_word",
        "line_spacing": 1.1, "letter_spacing": 1.5,
        "size_ratio": 0.050, "margin_ratio": 0.11,
    },
    "MAGENTA": {
        "font": "Montserrat", "weight": 700, "italic": False, "char_width": 0.58,
        "text_color": "#FFFFFF", "highlight_color": "#FF2BD6",
        "outline_color": "#2B0A3D", "outline": 4.0,
        "shadow_color": "#000000", "shadow_radius": 1.5,
        "caps": False, "highlight_mode": "karaoke_word",
        "line_spacing": 1.1, "letter_spacing": 0.5,
        "size_ratio": 0.052, "margin_ratio": 0.11,
    },
    "GOLD": {
        "font": "Playfair Display", "weight": 700, "italic": False, "char_width": 0.52,
        "text_color": "#FFD700", "highlight_color": "#FFF3B0",
        "outline_color": "#221A02", "outline": 3.5,
        "shadow_color": "#000000", "shadow_radius": 2.5,
        "caps": False, "highlight_mode": "karaoke_word",
        "line_spacing": 1.1, "letter_spacing": 0.0,
        "size_ratio": 0.048, "margin_ratio": 0.11,
    },
    "CRIMSON": {
        "font": "Oswald", "weight": 600, "italic": False, "char_width": 0.46,
        "text_color": "#FFFFFF", "highlight_color": "#DC143C",
        "outline_color": "#120404", "outline": 4.0,
        "shadow_color": "#000000", "shadow_radius": 1.5,
        "caps": True, "highlight_mode": "karaoke_word",
        "line_spacing": 1.05, "letter_spacing": 0.8,
        "size_ratio": 0.055, "margin_ratio": 0.12,
    },
    "CYBER": {
        "font": "Share Tech Mono", "weight": 400, "italic": False, "char_width": 0.60,
        "text_color": "#00FFC8", "highlight_color": "#FF073A",
        "outline_color": "#05141F", "outline": 6.0,
        "shadow_color": "#000000", "shadow_radius": 2.0,
        "caps": True, "highlight_mode": "box",
        "line_spacing": 1.1, "letter_spacing": 1.2,
        "size_ratio": 0.046, "margin_ratio": 0.11,
    },
    "BEAST": {
        "font": "Anton", "weight": 400, "italic": False, "char_width": 0.50,
        "text_color": "#FFFFFF", "highlight_color": "#FFE600",
        "outline_color": "#E62117", "outline": 8.0,
        "shadow_color": "#000000", "shadow_radius": 0.0,
        "caps": True, "highlight_mode": "box",
        "line_spacing": 1.0, "letter_spacing": 1.0,
        "size_ratio": 0.058, "margin_ratio": 0.13,
    },
    "MARKER": {
        "font": "Permanent Marker", "weight": 400, "italic": False, "char_width": 0.56,
        "text_color": "#FFE135", "highlight_color": "#FFFFFF",
        "outline_color": "#000000", "outline": 4.5,
        "shadow_color": "#000000", "shadow_radius": 0.0,
        "caps": False, "highlight_mode": "karaoke_word",
        "line_spacing": 1.15, "letter_spacing": 0.0,
        "size_ratio": 0.052, "margin_ratio": 0.12,
    },
    "RETRO": {
        "font": "Bungee", "weight": 400, "italic": False, "char_width": 0.62,
        "text_color": "#F5E9CF", "highlight_color": "#FF6F61",
        "outline_color": "#3E2723", "outline": 4.0,
        "shadow_color": "#000000", "shadow_radius": 2.0,
        "caps": True, "highlight_mode": "karaoke_word",
        "line_spacing": 1.05, "letter_spacing": 1.5,
        "size_ratio": 0.048, "margin_ratio": 0.12,
    },
    "GLITCH": {
        "font": "Orbitron", "weight": 700, "italic": False, "char_width": 0.62,
        "text_color": "#FFFFFF", "highlight_color": "#FF2BD6",
        "outline_color": "#00FFF7", "outline": 3.0,
        "shadow_color": "#FF073A", "shadow_radius": 3.5,
        "caps": True, "highlight_mode": "karaoke_word",
        "line_spacing": 1.05, "letter_spacing": 1.0,
        "size_ratio": 0.050, "margin_ratio": 0.12,
    },
    "MINIMAL": {
        "font": "Inter", "weight": 400, "italic": False, "char_width": 0.52,
        "text_color": "#FFFFFF", "highlight_color": "#FFFFFF",
        "outline_color": "#000000", "outline": 1.2,
        "shadow_color": "#000000", "shadow_radius": 1.0,
        "caps": False, "highlight_mode": "none",
        "line_spacing": 1.25, "letter_spacing": 0.0,
        "size_ratio": 0.040, "margin_ratio": 0.09,
    },
    "EMERALD": {
        "font": "Lato", "weight": 700, "italic": False, "char_width": 0.52,
        "text_color": "#FFFFFF", "highlight_color": "#10B981",
        "outline_color": "#06251B", "outline": 3.5,
        "shadow_color": "#000000", "shadow_radius": 1.5,
        "caps": False, "highlight_mode": "karaoke_word",
        "line_spacing": 1.1, "letter_spacing": 0.3,
        "size_ratio": 0.050, "margin_ratio": 0.11,
    },
    "SUNSET": {
        "font": "Poppins", "weight": 600, "italic": False, "char_width": 0.54,
        "text_color": "#FFF7E6", "highlight_color": "#FF7E5F",
        "outline_color": "#2D1B4E", "outline": 4.0,
        "shadow_color": "#2D1B4E", "shadow_radius": 2.0,
        "caps": False, "highlight_mode": "karaoke_word",
        "line_spacing": 1.1, "letter_spacing": 0.3,
        "size_ratio": 0.050, "margin_ratio": 0.11,
    },
    "VIOLET": {
        "font": "Quicksand", "weight": 600, "italic": False, "char_width": 0.54,
        "text_color": "#FFFFFF", "highlight_color": "#A78BFA",
        "outline_color": "#2E1065", "outline": 3.5,
        "shadow_color": "#000000", "shadow_radius": 1.5,
        "caps": False, "highlight_mode": "karaoke_word",
        "line_spacing": 1.15, "letter_spacing": 0.5,
        "size_ratio": 0.050, "margin_ratio": 0.11,
    },
}


def hex_to_ass(hex_color, alpha="00"):
    value = hex_color.lstrip("#")
    if len(value) == 3:
        value = "".join(ch * 2 for ch in value)
    r, g, b = value[0:2], value[2:4], value[4:6]
    return f"&H{alpha}{b}{g}{r}".upper()


def fmt_ass_time(seconds):
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"{hours:d}:{minutes:02d}:{seconds % 60:05.2f}"


def generate_subtitles(params: dict, render_hints: dict, clips_final: list,
                        transcript: dict, subs_dir) -> dict:
    """Build per-clip ASS files + the subtitles.json manifest (port of PY_ASS)."""
    style = params["subtitle_style"]
    geometry = render_hints["geometry"]
    out_w, out_h = geometry["out_w"], geometry["out_h"]
    language = params["subtitle_language"] or transcript.get("language") or ""
    subs_dir.mkdir(parents=True, exist_ok=True)

    if style == "NONE":
        subtitles = {"style": "NONE", "language": language, "word_timings_used": False, "clips": []}
        logger.info("subtitle_style is NONE: no subtitles will be generated or burned in.")
        return subtitles

    preset = SUBTITLE_PRESETS[style]
    scale = out_h / 1080.0
    fontsize = max(20, min(160, round(preset["size_ratio"] * out_h)))
    max_chars = max(10, int((out_w * 0.9) / (fontsize * preset["char_width"])))
    outline_px = max(1, round(preset["outline"] * scale)) if preset["outline"] else 0
    shadow_px = round(preset["shadow_radius"] * 0.45 * scale) if preset["shadow_radius"] else 0
    blur_px = round(preset["shadow_radius"] * scale) if preset["shadow_radius"] else 0
    spacing_px = round(preset["letter_spacing"] * scale)
    bold = 1 if preset["weight"] >= 600 else 0
    italic = 1 if preset["italic"] else 0
    highlight_mode = preset["highlight_mode"]
    caps = preset["caps"]

    if highlight_mode == "none":
        primary = hex_to_ass(preset["text_color"])
        secondary = hex_to_ass(preset["text_color"])
    else:
        primary = hex_to_ass(preset["highlight_color"])
        secondary = hex_to_ass(preset["text_color"])
    outline_colour = hex_to_ass(preset["outline_color"])
    border_style = 3 if highlight_mode == "box" else 1
    if preset["shadow_radius"] > 0:
        back_colour = hex_to_ass(preset["shadow_color"], alpha="70")
    else:
        back_colour = hex_to_ass("#000000", alpha="FF")

    use_karaoke = highlight_mode != "none" and transcript.get("word_timings_available")
    word_source = "whisper" if transcript.get("word_timings_available") else "whisper_segments"

    def build_word_cues(words):
        cues = []
        current = []
        for word in words:
            text = word["word"].strip()
            if not text:
                continue
            if current:
                gap = word["start"] - current[-1]["end"]
                duration = word["end"] - current[0]["start"]
                joined = len(" ".join(w["word"] for w in current)) + len(text) + 1
                ends_sentence = current[-1]["word"].rstrip().endswith((".", "!", "?"))
                if (gap > 0.6 or len(current) >= 8 or duration > 3.2
                        or joined > 2 * max_chars or (ends_sentence and gap > 0.25)):
                    cues.append(current)
                    current = []
            current.append(word)
        if current:
            cues.append(current)
        return cues

    def split_index_for(words):
        text = " ".join(w["word"] for w in words)
        if len(text) <= max_chars or len(words) == 1:
            return None
        best = None
        for i in range(1, len(words)):
            first = " ".join(w["word"] for w in words[:i])
            second = " ".join(w["word"] for w in words[i:])
            if len(first) <= max_chars and len(second) <= max_chars:
                imbalance = abs(len(first) - len(second))
                if best is None or imbalance < best[0]:
                    best = (imbalance, i)
        if best is not None:
            return best[1]
        return max(1, len(words) // 2)

    def dialogue_text(cue_words, split_at):
        parts = []
        for i, word in enumerate(cue_words):
            text = word["word"]
            if caps:
                text = text.upper()
            if use_karaoke:
                if i < len(cue_words) - 1:
                    dur = max(4, int(round((cue_words[i + 1]["start"] - word["start"]) * 100)))
                else:
                    dur = max(4, int(round(max(0.08, word["end"] - word["start"]) * 100)))
                tag = f"\\k{dur}"
                if i == 0 and blur_px > 0:
                    tag = f"\\blur{blur_px}" + tag
                token = "{" + tag + "}" + text
            else:
                if i == 0 and blur_px > 0:
                    token = "{\\blur" + str(blur_px) + "}" + text
                else:
                    token = text
            if parts:
                parts.append(" ")
            parts.append(token)
            if split_at is not None and i == split_at - 1 and split_at < len(cue_words):
                parts.append("{\\N}")
        return "".join(parts)

    subtitle_clips = []
    for clip in clips_final:
        index = clip["clip_index"]
        start = clip["start_sec"]
        end = clip["end_sec"]
        length = round(end - start, 3)
        band = (render_hints["clips"].get(str(index)) or {}).get("caption_band", "bottom")
        alignment = 8 if band == "top" else 2
        margin_v = round(out_h * preset["margin_ratio"])
        margin_lr = round(out_w * 0.05)

        words = []
        if transcript.get("word_timings_available"):
            for word in transcript.get("words", []):
                if word["start"] < end and word["end"] > start:
                    words.append({
                        "word": word["word"],
                        "start": max(word["start"], start),
                        "end": min(word["end"], end),
                    })
            words.sort(key=lambda w: w["start"])

        cues = []
        if words:
            for cue_words in build_word_cues(words):
                cue_start = max(0.0, cue_words[0]["start"] - start)
                cue_end = min(length, cue_words[-1]["end"] - start + 0.25)
                if cue_end - cue_start < 0.6:
                    cue_end = min(length, cue_start + 0.6)
                cues.append({"start": round(cue_start, 2), "end": round(cue_end, 2),
                             "words": cue_words})
        else:
            # Segment-level timing only: no fabricated word synchronization.
            for segment in transcript.get("segments", []):
                if segment["end"] <= start or segment["start"] >= end:
                    continue
                text = segment["text"].strip()
                if not text:
                    continue
                seg_start = max(segment["start"], start) - start
                seg_end = min(segment["end"], end) - start
                chunks = []
                current = []
                for token in text.split():
                    candidate = " ".join(current + [token])
                    if current and len(candidate) > 2 * max_chars:
                        chunks.append(" ".join(current))
                        current = [token]
                    else:
                        current.append(token)
                if current:
                    chunks.append(" ".join(current))
                total_chars = sum(len(c) for c in chunks) or 1
                cursor = seg_start
                span = max(0.6, seg_end - seg_start)
                for chunk in chunks:
                    share = span * len(chunk) / total_chars
                    cues.append({"start": round(max(0.0, cursor), 2),
                                 "end": round(min(length, cursor + share), 2), "words": None,
                                 "text": chunk})
                    cursor += share

        # Resolve overlaps: captions never stack.
        cues.sort(key=lambda c: c["start"])
        for i in range(1, len(cues)):
            if cues[i]["start"] < cues[i - 1]["end"] + 0.05:
                if cues[i - 1]["end"] - cues[i - 1]["start"] > 0.8:
                    cues[i - 1]["end"] = round(max(cues[i]["start"] - 0.05, cues[i - 1]["start"] + 0.5), 2)
                else:
                    cues[i]["start"] = round(cues[i - 1]["end"] + 0.05, 2)

        events = []
        plain_cues = []
        for cue in cues:
            if cue.get("words"):
                split_at = split_index_for(cue["words"])
                text = dialogue_text(cue["words"], split_at)
                plain = " ".join(w["word"] for w in cue["words"])
            else:
                chunk = cue.get("text", "")
                lines = [chunk] if len(chunk) <= max_chars else None
                if lines is None:
                    tokens = chunk.split()
                    best = None
                    for i in range(1, len(tokens)):
                        first = " ".join(tokens[:i])
                        second = " ".join(tokens[i:])
                        if len(first) <= max_chars and len(second) <= max_chars:
                            imbalance = abs(len(first) - len(second))
                            if best is None or imbalance < best[0]:
                                best = (imbalance, first, second)
                    if best:
                        lines = [best[1], best[2]]
                    else:
                        midpoint = len(chunk) // 2
                        cut = chunk.rfind(" ", 0, midpoint)
                        cut = cut if cut > 0 else midpoint
                        lines = [chunk[:cut].strip(), chunk[cut:].strip()]
                text = "{\\N}".join(lines)
                if blur_px > 0:
                    text = "{\\blur" + str(blur_px) + "}" + text
                plain = chunk
            events.append(f"Dialogue: 0,{fmt_ass_time(cue['start'])},{fmt_ass_time(cue['end'])},"
                          f"Clipzy,,0,0,0,,{text}")
            plain_cues.append({"start": cue["start"], "end": cue["end"], "text": plain})

        burned_in = bool(events)
        ass_path = subs_dir / f"clip_{index}.ass"
        if burned_in:
            header = (
                "[Script Info]\n"
                f"; Clipzy preset {style} (mirrors Android SubtitleStyles.kt)\n"
                "ScriptType: v4.00+\n"
                f"PlayResX: {out_w}\n"
                f"PlayResY: {out_h}\n"
                "WrapStyle: 0\n"
                "ScaledBorderAndShadow: yes\n"
                "\n"
                "[V4+ Styles]\n"
                "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
                "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
                "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
                f"Style: Clipzy,{preset['font']},{fontsize},{primary},{secondary},{outline_colour},"
                f"{back_colour},{bold},{italic},0,0,100,100,{spacing_px},0,{border_style},"
                f"{outline_px},{shadow_px},{alignment},{margin_lr},{margin_lr},{margin_v},1\n"
                "\n"
                "[Events]\n"
                "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
            )
            ass_path.write_text(header + "\n".join(events) + "\n", encoding="utf-8")

        subtitle_clips.append({
            "clip_index": index,
            "burned_in": burned_in,
            "file": str(ass_path) if burned_in else None,
            "alignment": alignment,
            "cues": plain_cues,
        })
        if burned_in:
            logger.info("Clip %d: %d caption events (%s).", index, len(events),
                        "word-synced karaoke" if use_karaoke else "segment-level timing")
        else:
            logger.info("Clip %d: no speech in range; nothing burned in (reported honestly).", index)

    subtitles = {
        "style": style,
        "language": language,
        "word_timings_used": use_karaoke,
        "word_source": word_source,
        "preset": {
            "font": preset["font"], "fontsize": fontsize, "bold": bool(bold),
            "italic": bool(italic), "highlight_mode": highlight_mode, "caps": caps,
            "outline_px": outline_px, "shadow_px": shadow_px, "spacing_px": spacing_px,
        },
        "clips": subtitle_clips,
    }
    logger.info("ASS subtitles generated for preset %s on %dx%d canvas (fontsize %d, karaoke=%s).",
                style, out_w, out_h, fontsize, use_karaoke)
    return subtitles
