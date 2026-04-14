PYTHON := python

.PHONY: setup audit train test

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

audit:
	$(PYTHON) main.py audit --data data/raw/clintox.csv

train:
	$(PYTHON) main.py train --data data/raw/clintox.csv --config configs/defaults.json

test:
	pytest
