.PHONY: help install train analysis predict test lint run docker-build docker-run clean

help:
	@echo "install       - установить зависимости"
	@echo "train         - обучить классические модели"
	@echo "analysis      - графики и анализ ошибок"
	@echo "predict       - пример инференса"
	@echo "test          - запустить тесты"
	@echo "lint          - flake8"
	@echo "run           - Streamlit локально"
	@echo "docker-build  - собрать образ"
	@echo "docker-run    - запустить контейнер"

install:
	pip install -r requirements.txt

train:
	python -m src.train

analysis:
	python -m src.analysis

predict:
	python -m src.predict

test:
	pytest tests/ -v

lint:
	flake8 src/ app.py --max-line-length=120

run:
	streamlit run app.py

docker-build:
	docker build -t spam-classifier .

docker-run:
	docker run --rm -p 8501:8501 spam-classifier

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
