"""End-to-end integration test (GPT-16). Runs the real pipeline (LM Studio,
SD1.5, Piper, ffmpeg) on the short sample rhyme - skipped automatically if
LM Studio isn't reachable, since this needs the full local stack, not
mocks. Slow (minutes on a cold run; seconds if a prior run's outputs are
still cached, since orchestration resumes by default)."""
from pathlib import Path

import pytest

from src.config import load_config
from src.orchestration import run_pipeline
from src.utils.ffmpeg_helper import probe
from tests.conftest import requires_lm_studio

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_RHYME = REPO_ROOT / "data" / "rhyme.txt"


@pytest.mark.slow
@requires_lm_studio
def test_full_pipeline_produces_a_playable_video():
    assert SAMPLE_RHYME.exists(), "sample rhyme fixture is missing"

    output = run_pipeline(SAMPLE_RHYME, name="rhyme", force=False)

    assert output.exists()

    config = load_config()
    assert (config.paths["scripts_dir"] / "rhyme.json").exists()
    assert (config.paths["images_dir"] / "manifest.json").exists()
    assert (config.paths["audio_dir"] / "manifest.json").exists()

    info = probe(output)
    video_streams = [s for s in info["streams"] if s["codec_type"] == "video"]
    audio_streams = [s for s in info["streams"] if s["codec_type"] == "audio"]
    assert video_streams and video_streams[0]["codec_name"] == "h264"
    assert audio_streams
    assert float(info["format"]["duration"]) > 0


def test_stale_manifest_from_a_different_run_is_not_reused(tmp_path):
    """Per-scene manifests are not namespaced by run name, so a 16-scene run
    followed by a 4-scene one used to fail with "manifest has 16 entries,
    expected 4" and no way forward but deleting files by hand."""
    from src.orchestration import manifest_is_current
    from src.utils.file_manager import safe_write_json

    path = tmp_path / "manifest.json"
    safe_write_json(path, [{"scene_index": i} for i in range(16)])
    assert not manifest_is_current(path, 4)
    assert manifest_is_current(path, 16)
    assert not manifest_is_current(tmp_path / "missing.json", 4)


def test_corrupt_manifest_is_treated_as_stale(tmp_path):
    from src.orchestration import manifest_is_current

    path = tmp_path / "manifest.json"
    path.write_text("{not json", encoding="utf-8")
    assert not manifest_is_current(path, 4)
