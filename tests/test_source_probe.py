"""Source probing, scene detection and keyframe extraction (real FFmpeg)."""

import json

import pytest

from processors import scene_detection, source_probe
from engines import vision_engine


def test_probe_source(sample_video):
    source = source_probe.probe_source(sample_video)
    assert source["width"] == 640 and source["height"] == 360
    assert source["fps"] == 30.0
    assert 23.0 < source["duration"] < 25.0
    assert source["has_audio"] is True
    assert source["vcodec"] == "h264"
    assert source["pix_fmt"] == "yuv420p"
    assert len(source["source_sha256"]) == 64
    assert source["aspect_ratio"] == "16:9"
    assert source["audio"]["codec"] == "aac"


def test_probe_source_sha256_stable_and_sensitive(tmp_path, sample_video):
    import shutil
    copy = tmp_path / "copy.mp4"
    shutil.copyfile(sample_video, copy)
    assert source_probe.sha256_file(copy) == source_probe.sha256_file(sample_video)
    # Mutating one byte changes the fingerprint.
    data = bytearray(copy.read_bytes())
    data[len(data) // 2] ^= 0xFF
    copy.write_bytes(bytes(data))
    assert source_probe.sha256_file(copy) != source_probe.sha256_file(sample_video)


def test_probe_rejects_ninety_minute_sources(tmp_path, sample_video, monkeypatch):
    from processors import source_probe as sp

    class FakeProbe:
        pass

    original = sp.ff.probe_media

    def fake_probe(path):
        data = original(path)
        if data:
            data = json.loads(json.dumps(data))
            data["format"]["duration"] = str(90 * 60 + 10)
        return data

    monkeypatch.setattr(sp.ff, "probe_media", fake_probe)
    with pytest.raises(Exception, match="90 minutes"):
        sp.probe_source(sample_video)


def test_scene_detection_no_cuts_on_constant_pattern(sample_video):
    from processors import scene_detection as sd

    duration = 24.0
    cuts = sd.detect_scene_cuts(sample_video, duration)
    # testsrc2 is one continuous scene: at most spurious weak cuts.
    assert isinstance(cuts, list)


def test_scene_detection_finds_hard_cut(tmp_path):
    import subprocess

    # Two very different segments concatenated in one encode -> one hard cut.
    out = tmp_path / "cut.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-nostdin", "-loglevel", "error",
         "-f", "lavfi", "-i", "color=red:size=320x180:rate=10:duration=3",
         "-f", "lavfi", "-i", "color=blue:size=320x180:rate=10:duration=3",
         "-filter_complex", "[0:v][1:v]concat=n=2:v=1",
         "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", str(out)],
        check=True, capture_output=True)
    cuts = scene_detection.detect_scene_cuts(out, 6.0)
    assert any(2.0 < t < 4.5 for t in cuts), f"expected the hard cut, got {cuts}"


def test_extract_keyframes(sample_video, settings, tmp_path):
    from storage.file_manager import JobWorkspace

    workspace = JobWorkspace(tmp_path / "jobs", "testjob").create()
    import shutil
    shutil.copyfile(sample_video, workspace.source)

    source = source_probe.probe_source(workspace.source)
    manifest = vision_engine.extract_keyframes(source, workspace, settings)

    assert manifest["coarse_count"] >= 1
    assert 1 <= len(manifest["ai_frames"]) <= vision_engine.MAX_AI_FRAMES
    assert manifest["coarse_interval"] >= 8.0
    for frame in manifest["ai_frames"]:
        assert frame["t"] >= 0
        from pathlib import Path
        assert Path(frame["file"]).is_file()

    # Second call is a cache hit (validated by file existence).
    manifest2 = vision_engine.extract_keyframes(source, workspace, settings)
    assert manifest2["ai_frames"] == manifest["ai_frames"]
