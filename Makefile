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

# Removes every generated pipeline artifact and Python cache, leaving a tree
# that is ready for a clean end-to-end run. Does NOT touch input rhymes, the
# character bank, model weights, or the venv.
#
# data/characters holds the IP-Adapter reference portraits. They ARE regenerated
# on demand, but deleting them changes what every character looks like from then
# on - so this target leaves them alone. Use `clean-characters` to reset
# appearances deliberately.
#
# `git clean -X` is deliberately not used here: it would also remove the venv
# and model weights, which are gitignored but expensive to rebuild.
clean:
	@# Generated scripts only. data/scripts/rhyme.json is a tracked sample that
	@# ships with the repo, so a blanket *.json deleted a checked-in file.
	find data/scripts -name '*.json' ! -name 'rhyme.json' -delete 2>/dev/null || true
	rm -rf data/images data/audio data/motion data/output data/images_noip data/generated \
	       logs/*.log .pytest_cache
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true

# Forget every character's established appearance. The next run re-registers
# them from the poem and generates new reference portraits.
clean-characters:
	rm -rf data/characters
