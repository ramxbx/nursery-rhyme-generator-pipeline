import json
import time
import urllib.request
from pathlib import Path

ENDPOINT = "http://127.0.0.1:1234/v1/chat/completions"
MODEL = "qwen2.5-7b-bench"

SYSTEM = (
    "You are a creative director for a children's animated video. Given one "
    "scene from a nursery rhyme script, write a long, precise, vivid scene "
    "description (120-180 words) that a text-to-image artist AND a voice "
    "director could both use. Cover, in prose (not a list): the setting and "
    "environment, lighting and color palette, the main character's precise "
    "appearance and pose/action, background details, and the emotional tone "
    "of the moment (which should also guide how the line is narrated - pace, "
    "warmth, energy). Be concrete and specific, not generic. Output ONLY the "
    "description paragraph, no headers, no preamble."
)

REPO_ROOT = Path(__file__).resolve().parent.parent
script = json.loads((REPO_ROOT / "data" / "scripts" / "rhyme.json").read_text(encoding="utf-8"))

for i, scene in enumerate(script["scenes"], start=1):
    user = (
        f"Line: \"{scene['line']}\"\n"
        f"Speaker: {scene['speaker']}\n"
        f"Current short stage direction: {scene['stage_direction']}\n"
        f"Current short scene description: {scene['scene_description']}\n"
        f"Current mood tag: {scene['mood']}"
    )
    payload = {"model": MODEL, "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
               "temperature": 0.7, "max_tokens": 400}
    req = urllib.request.Request(ENDPOINT, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=180) as resp:
        body = json.loads(resp.read().decode())
    elapsed = time.perf_counter() - start
    text = body["choices"][0]["message"]["content"].strip()
    usage = body.get("usage", {})
    print(f"=== Scene {i}: \"{scene['line']}\" ({elapsed:.1f}s, {usage.get('completion_tokens')} tokens) ===")
    print(text)
    print()
