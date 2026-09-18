import streamlit as st
from src.predict import load_artifacts, predict

st.set_page_config(
    page_title="Spam Detector",
    page_icon="📧",
    layout="centered",
)

st.title(" Spam Detector")
st.caption("TF-IDF + классический ML. LSTM-подход — в ноутбуке `notebooks/02_lstm.ipynb`.")


@st.cache_resource
def get_artifacts():
    return load_artifacts()


model, vectorizer = get_artifacts()

text = st.text_area(
    "Введите текст сообщения:",
    height=200,
    placeholder="Subject: win a free iPhone now! Click here to claim your prize.",
)

col1, col2 = st.columns([1, 3])
with col1:
    predict_clicked = st.button("Проверить", type="primary", use_container_width=True)
with col2:
    clear_clicked = st.button("Очистить", use_container_width=True)

if clear_clicked:
    st.rerun()

if predict_clicked:
    if text.strip():
        with st.spinner("Анализирую..."):
            result = predict(text, model, vectorizer)
        proba = result["probability"] or 0.0
        if result["label"] == 1:
            st.error(f" СПАМ — вероятность {proba:.1%}")
        else:
            st.success(f" НЕ СПАМ — вероятность спама {proba:.1%}")

        with st.expander("Детали"):
            st.write(f"**Label:** {result['label']}")
            st.write(f"**Probability:** {proba:.4f}")
    else:
        st.warning("Введите текст сообщения.")

with st.expander("О проекте"):
    st.markdown(
        """
        - **Датасет**: [Spam Email Dataset](https://www.kaggle.com/karthickveerakumar/spam-filter) — 5.7k писем, дисбаланс 3.2:1.
        - **Модели**: LogisticRegression, RandomForest, LinearSVC, XGBoost (TF-IDF) + BiLSTM.
        - **Метрики**: F1 ≈ 0.97, ROC-AUC ≈ 0.99 (см. README).
        - **Код**: [GitHub](https://github.com/SiniauskiArtsiom/Spam-detector)
        """
    )
