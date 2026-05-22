PYTHON ?= python3

.PHONY: format lint test check

format:
	bash scripts/format.sh

lint:
	bash scripts/lint.sh

test:
	$(PYTHON) -m pytest

check: lint test
