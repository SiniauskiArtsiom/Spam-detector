# Spam Classifier

Классификация спам-сообщений: классический ML (TF-IDF + LogReg/RF/LinearSVC/XGBoost) и LSTM.

## Датасет
[Spam Email Dataset на Kaggle](https://www.kaggle.com/karthickveerakumar/spam-filter)

## Результаты
| Модель | F1 (spam) | ROC-AUC |
|--------|-----------|---------|
| ...    | ...       | ...     |

## Структура
- `src/` — исходный код
- `notebooks/` — EDA и LSTM-эксперименты
- `models/` — сохранённые модели
- `tests/` — unit-тесты
- `app.py` — Streamlit демо

## Запуск
```bash
pip install -r requirements.txt
python -m src.train
streamlit run app.py
