"""End-to-end integration test (GPT-16). Runs the real pipeline (LM Studio,
SD1.5, Piper, ffmpeg) on the short sample rhyme - skipped automatically if
LM Studio isn't reachable, since this needs the full local stack, not
mocks. Slow (minutes on a cold run; seconds if a prior run's outputs are
still cached, since orchestration resumes by default)."""
from pathlib import Path

from src.config import load_config
from src.orchestration import run_pipeline
from src.utils.ffmpeg_helper import probe
from tests.conftest import requires_lm_studio

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_RHYME = REPO_ROOT / "data" / "rhyme.txt"


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
