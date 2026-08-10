"""Smoke test for GPT-6/7/8/9 infra utils (formal suite lands in GPT-16)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.utils.file_manager import ensure_dirs, safe_write_json, read_json, scene_path, clean_dir
from src.utils.logger import get_logger, log_with_fields
from src.utils.prompt_builder import build_dialogue_prompt, build_scene_image_prompt

if __name__ == "__main__":
    cfg = load_config()
    assert cfg.llm["primary"] == "lfm2-1.2b-bench"
    assert cfg.video["width"] == 1920
    print("[config] OK:", cfg.llm)

    dirs = ensure_dirs(cfg.paths)
    test_dir = dirs["output_dir"] / "_smoke_test"
    ensure_dirs({"t": test_dir})
    p = scene_path(test_dir, 2, ".json")
    safe_write_json(p, {"hello": "world"})
    assert read_json(p) == {"hello": "world"}
    clean_dir(test_dir, keep_dir=False)
    print("[file_manager] OK:", p.name)

    logger = get_logger("smoke_test")
    log_with_fields(logger, 20, "smoke test log line", scene=1, latency_s=0.42)
    print("[logger] OK")

    dp = build_dialogue_prompt("Twinkle, twinkle, little star,", 1, 4, [])
    assert "Twinkle, twinkle, little star," in dp
    sp = build_scene_image_prompt("Star", "a friendly cartoon star with a smiling face", "shining in a night sky")
    assert "shining in a night sky" in sp
    print("[prompt_builder] OK")

    print("ALL INFRA SMOKE TESTS PASSED")
