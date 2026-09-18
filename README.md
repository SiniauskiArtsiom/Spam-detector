# Spam Detector

![CI](https://github.com/SiniauskiArtsiom/Spam-detector/actions/workflows/ci.yml/badge.svg)

Классификация спам-сообщений. Сравнение классического ML (TF-IDF + LogReg/RF/LinearSVC/XGBoost) и BiLSTM.

## Датасет

[Spam Email Dataset на Kaggle](https://www.kaggle.com/karthickveerakumar/spam-filter) — 5 728 писем, дисбаланс 3.2:1 (ham/spam). Признак `text`, целевая переменная `spam` (0/1).

## Подход

- **Предобработка**: нижний регистр, удаление пунктуации и стоп-слов, лемматизация (NLTK).
- **Признаки**: TF-IDF, 5000 признаков, unigrams + bigrams.
- **Классический ML**: LogisticRegression, RandomForest, LinearSVC, XGBoost — подбор гиперпараметров через GridSearchCV (3-fold, scoring = F1).
- **DL**: BiLSTM (Embedding → SpatialDropout → BiLSTM → Dense → Dropout → Sigmoid). См. [`notebooks/02_lstm.ipynb`](notebooks/02_lstm.ipynb).

## Результаты

| Модель | Precision | Recall | F1 | ROC-AUC |
|--------|-----------|--------|-----|---------|
| LogReg | 0.9713 | 0.989 | 0.98 | 0.9994 |
| RandomForest | 0.9815 | 0.9745 | 0.9761 | 0.9988 |
| LinearSVC | 0.978 | 0.9745 | 0.9762 | 0.9995 |
| XGBoost | 0.9464 | 0.9672 | 0.9567 | 0.9974 |
| **BiLSTM** | 0.9931 | 0.9908 | 0.9745 | 0.9984 |




## Быстрый старт

```bash
# 1. Установка
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Обучение (сравнит 4 модели, выберет лучшую)
make train

# 3. Графики и анализ ошибок
make analysis

# 4. Локальное демо
make run      # http://localhost:8501

# 5. Тесты и линтер
make test
make lint
