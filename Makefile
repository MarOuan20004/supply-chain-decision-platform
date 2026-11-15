.PHONY: install db-init baselines train test benchmark lint format clean

PYTHON ?= python3

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev,agent]"

db-init:
	$(PYTHON) -m supply_chain.data_pipeline --init-db

baselines:
	$(PYTHON) scripts/run_baselines_mlflow.py

train:
	$(PYTHON) scripts/train_and_register_mlflow.py

benchmark:
	$(PYTHON) benchmarks/evaluate_agent_vs_baselines.py

test:
	pytest tests/ -v

lint:
	ruff check src/ scripts/ benchmarks/ tests/
	mypy src/

format:
	ruff format src/ scripts/ benchmarks/ tests/

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov/ catboost_info/
