.PHONY: install dev test eval run lint clean

install:
	python -m pip install -e .

dev:
	python -m pip install -e ".[dev]"

test:
	pytest -v

run:
	python -m research_assistant.main "$(Q)"

eval:
	python -m evals.harness

eval-fixture:
	SEARCH_MODE=fixture python -m evals.harness

lint:
	ruff check src evals tests

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
