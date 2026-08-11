import json
import time
import urllib.request

ENDPOINT = "http://127.0.0.1:1234/v1/chat/completions"
MODELS = ["lfm2-1.2b-bench", "gemma-3-1b-bench"]

SYSTEM = (
    "You are a children's poet. Rewrite the given nursery rhyme as a NEW, "
    "original poem: keep the same number of lines, the same rhyme scheme "
    "(which lines rhyme with which), and a similar playful theme, but use "
    "different, fun, original wording - do not reuse the source lines "
    "verbatim. Output ONLY the rewritten poem, one line per line, no "
    "preamble, no numbering."
)
RHYME = (
    "Twinkle, twinkle, little star,\n"
    "How I wonder what you are.\n"
    "Up above the world so high,\n"
    "Like a diamond in the sky."
)

for model in MODELS:
    payload = {"model": model, "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": RHYME}],
               "temperature": 0.7, "max_tokens": 200}
    req = urllib.request.Request(ENDPOINT, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = json.loads(resp.read().decode())
    elapsed = time.perf_counter() - start
    text = body["choices"][0]["message"]["content"]
    print(f"=== {model} ({elapsed:.2f}s) ===")
    print(text)
    print()
