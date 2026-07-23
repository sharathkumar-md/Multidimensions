.PHONY: venv install dev lint test clean

PYTHON  := python3.11
VENV    := .venv
PIP     := $(VENV)/bin/pip
UVICORN := $(VENV)/bin/uvicorn

# ── Setup ─────────────────────────────────────────────────────────────────────

venv:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip

install: venv
	$(PIP) install -r requirements.txt

# ── Dev server ────────────────────────────────────────────────────────────────

dev: install
	API_RELOAD=true $(UVICORN) main:app --host 0.0.0.0 --port 8000 --reload

dev-no-reload:
	$(UVICORN) main:app --host 0.0.0.0 --port 8000

# ── Lint ──────────────────────────────────────────────────────────────────────

lint:
	$(VENV)/bin/ruff check . --fix
	$(VENV)/bin/ruff format .

# ── Clean ─────────────────────────────────────────────────────────────────────

clean:
	rm -rf $(VENV) __pycache__ .sessions.json *.pyc
