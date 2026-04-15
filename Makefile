PYTHON := python

.PHONY: setup run audit train test

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

run:
	$(PYTHON) run_end_to_end.py

audit:
	$(PYTHON) main.py audit --data data/raw/clintox.csv

train:
	$(PYTHON) main.py train --data data/raw/clintox.csv --config configs/defaults.json

test:
	pytest
