PYTHON ?= python
VENV := .venv
VENV_PYTHON := $(VENV)/Scripts/python.exe
VENV_PIP := $(VENV)/Scripts/pip.exe
TORCH_INDEX := https://download.pytorch.org/whl/cu126
INPUT ?= data/rhyme.txt

.PHONY: setup install run test clean

# First-time bootstrap: create the venv and install everything.
setup: $(VENV_PYTHON) install

$(VENV_PYTHON):
	$(PYTHON) -m venv $(VENV)

# Install/update Python dependencies into the existing venv. torch needs the
# CUDA-specific index (see requirements.txt header) rather than plain PyPI.
install: $(VENV_PYTHON)
	$(VENV_PIP) install --quiet --index-url $(TORCH_INDEX) torch
	$(VENV_PIP) install --quiet -r requirements.txt
	@echo "Python deps installed. Model weights and LM Studio setup are separate"
	@echo "one-time steps - see README.md."

# Run the full pipeline end-to-end. Override the input rhyme with:
#   make run INPUT=data/my_rhyme.txt
run: $(VENV_PYTHON)
	$(VENV_PYTHON) -m src.orchestration $(INPUT)

test: $(VENV_PYTHON)
	$(VENV_PYTHON) -m pytest tests/ -v

# Removes generated pipeline artifacts (scripts/images/audio/output/logs) and
# Python caches. Does NOT touch input files, model weights, or the venv.
clean:
	rm -rf data/scripts/*.json data/images/*.png data/images/manifest.json \
	       data/audio/*.wav data/audio/manifest.json data/output/*.mp4 \
	       logs/*.log .pytest_cache
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
