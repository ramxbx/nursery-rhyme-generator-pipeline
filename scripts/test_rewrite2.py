import json
import re
import time
import urllib.request

ENDPOINT = "http://127.0.0.1:1234/v1/chat/completions"
MODEL = "lfm2-1.2b-bench"

LINES = [
    "Twinkle, twinkle, little star,",
    "How I wonder what you are.",
    "Up above the world so high,",
    "Like a diamond in the sky.",
]

SYSTEM_TMPL = (
    "Rewrite this nursery rhyme line so it ends with the exact word \"{word}\" "
    "(keep that word, drop nothing, add no words after it except matching "
    "punctuation). Make everything before that word fun, playful, and "
    "DIFFERENT from the original wording - do not just copy the original. "
    "Keep it roughly the same length. Output ONLY the rewritten line, "
    "nothing else."
)


def call(system, user):
    payload = {"model": MODEL, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
               "temperature": 0.8, "max_tokens": 60}
    req = urllib.request.Request(ENDPOINT, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode())["choices"][0]["message"]["content"].strip()


for line in LINES:
    m = re.match(r"^(.*\b)([A-Za-z']+)([.,!?]*)$", line.strip())
    body, last_word, punct = m.group(1), m.group(2), m.group(3)
    system = SYSTEM_TMPL.format(word=last_word + punct)
    start = time.perf_counter()
    result = call(system, line)
    elapsed = time.perf_counter() - start
    ends_ok = result.rstrip().lower().endswith((last_word + punct).lower()) or result.rstrip().lower().endswith(last_word.lower())
    print(f"[{elapsed:.2f}s] target_end={last_word!r} ends_ok={ends_ok}")
    print(f"  orig: {line}")
    print(f"  new:  {result}")
