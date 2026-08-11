import sys
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _lm_studio_available() -> bool:
    try:
        urllib.request.urlopen("http://127.0.0.1:1234/v1/models", timeout=2)
        return True
    except Exception:
        return False


requires_lm_studio = pytest.mark.skipif(
    not _lm_studio_available(), reason="LM Studio endpoint not reachable at 127.0.0.1:1234"
)
