.PHONY: install run test
install:
	pip install -r requirements.txt
run:
	python run_forecast.py
test:
	pytest -q
