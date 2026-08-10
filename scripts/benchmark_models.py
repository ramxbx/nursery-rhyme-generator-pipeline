"""GPT-18 benchmark: structured-JSON compliance + latency for the three
CPU-hosted local model candidates, via LM Studio's OpenAI-compatible API.
"""
import json
import time
import urllib.request

ENDPOINT = "http://127.0.0.1:1234/v1/chat/completions"
MODELS = ["gemma-3-1b-bench", "lfm2-1.2b-bench"]

SYSTEM_PROMPT = (
    "You are a children's video script assistant. Convert the given nursery "
    "rhyme into a JSON object with this exact schema, and output ONLY the "
    "JSON, no other text:\n"
    '{"cast": ["<character name>", ...], '
    '"scenes": [{"line": "<verbatim source line>", "speaker": "<cast member>", '
    '"stage_direction": "<brief visual description>"}]}\n\n'
    "Rules: create exactly ONE scene object per line of the source rhyme, in "
    "the same order as the source, one scene per line — do not merge multiple "
    "source lines into a single scene. The \"cast\" array must contain only "
    "plain strings (character names), never objects."
)

RHYME = (
    "Twinkle, twinkle, little star,\n"
    "How I wonder what you are.\n"
    "Up above the world so high,\n"
    "Like a diamond in the sky."
)

REQUIRED_KEYS = {"cast", "scenes"}
SCENE_KEYS = {"line", "speaker", "stage_direction"}


def call_model(model: str) -> dict:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": RHYME},
        ],
        "temperature": 0.2,
        "max_tokens": 700,
    }
    req = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    elapsed = time.perf_counter() - start

    content = body["choices"][0]["message"]["content"]
    usage = body.get("usage", {})

    result = {
        "model": model,
        "latency_s": round(elapsed, 2),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "raw": content,
        "json_valid": False,
        "schema_valid": False,
        "lines_covered": None,
        "error": None,
    }

    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    try:
        parsed = json.loads(text)
        result["json_valid"] = True
    except Exception as e:
        result["error"] = f"json parse failed: {e}"
        return result

    if REQUIRED_KEYS.issubset(parsed.keys()) and isinstance(parsed.get("scenes"), list):
        scenes_ok = all(SCENE_KEYS.issubset(s.keys()) for s in parsed["scenes"] if isinstance(s, dict))
        result["schema_valid"] = scenes_ok
        result["lines_covered"] = len(parsed["scenes"])
    else:
        result["error"] = "missing required top-level keys"

    return result


if __name__ == "__main__":
    results = []
    for m in MODELS:
        print(f"== {m} ==")
        try:
            r = call_model(m)
        except Exception as e:
            r = {"model": m, "error": str(e), "json_valid": False, "schema_valid": False, "latency_s": None}
        print(json.dumps(r, indent=2)[:1200])
        results.append(r)

    print("\n=== SUMMARY ===")
    print(f"{'model':<20}{'latency_s':<12}{'json_valid':<12}{'schema_valid':<14}{'scenes':<8}")
    for r in results:
        print(f"{r['model']:<20}{str(r.get('latency_s')):<12}{str(r.get('json_valid')):<12}"
              f"{str(r.get('schema_valid')):<14}{str(r.get('lines_covered')):<8}")
