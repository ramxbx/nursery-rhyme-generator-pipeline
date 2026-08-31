import sys
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Keep test logging out of logs/pipeline.log.
#
# Every agent module calls get_logger() at import time, which attaches a
# FileHandler to the real log, so tests wrote thousands of lines into the same
# file a production run writes to - interleaved with it. Diagnosing a run then
# meant sifting out entries carrying pytest tmp paths, and a test's "character
# registered" line was indistinguishable from a real one.
#
# This has to happen here at module level rather than in a fixture: conftest is
# imported before the test modules that import the agents, and by fixture time
# the handler is already attached.
import tempfile  # noqa: E402

from src.utils import logger as _pipeline_logger  # noqa: E402

_pipeline_logger.DEFAULT_LOG_FILE = (
    Path(tempfile.mkdtemp(prefix="rhyme-tests-")) / "pipeline.log")


def _lm_studio_available() -> bool:
    try:
        urllib.request.urlopen("http://127.0.0.1:1234/v1/models", timeout=2)
        return True
    except Exception:
        return False


requires_lm_studio = pytest.mark.skipif(
    not _lm_studio_available(), reason="LM Studio endpoint not reachable at 127.0.0.1:1234"
)

