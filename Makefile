# LotIQ - common tasks. Run `make help` to list them.
# On Windows without `make`, just run the underlying commands (see README).

.PHONY: help setup data train snapshot pipeline api dashboard test clean

PYTHON ?= python

help:
	@echo "LotIQ tasks:"
	@echo "  make setup      Install the package and dependencies (editable)"
	@echo "  make data       Generate the synthetic chilli dataset"
	@echo "  make train      Train the risk model and write metrics"
	@echo "  make snapshot   Build the warehouse snapshot for the dashboard"
	@echo "  make pipeline   data + train + snapshot in one go"
	@echo "  make api        Run the FastAPI server (http://127.0.0.1:8000/docs)"
	@echo "  make dashboard  Run the Streamlit dashboard"
	@echo "  make test       Run the test suite"
	@echo "  make clean      Remove caches and build artefacts"

setup:
	$(PYTHON) -m pip install -e ".[dev]"

data:
	$(PYTHON) -m lotiq.data.generate

train:
	$(PYTHON) -m lotiq.models.train

snapshot:
	$(PYTHON) -m lotiq.data.snapshot

pipeline: data train snapshot

api:
	uvicorn api.main:app --reload

dashboard:
	streamlit run dashboard/app.py

test:
	pytest -q

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache **/__pycache__ *.egg-info build dist
	find . -name "*.pyc" -delete
