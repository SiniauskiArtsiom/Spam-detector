import joblib

from src.data import clean_text


def load_artifacts(model_path="models/best_model.joblib", vec_path="models/tfidf.joblib"):
    return joblib.load(model_path), joblib.load(vec_path)


def predict(text: str, model, vectorizer) -> dict:
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = int(model.predict(vec)[0])
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(vec)[0, 1])
    elif hasattr(model, "decision_function"):
        score = float(model.decision_function(vec)[0])
        proba = 1 / (1 + pow(2.718, -score))  # sigmoid
    else:
        proba = None
    return {"label": pred, "probability": proba}


if __name__ == "__main__":
    model, vec = load_artifacts()
    samples = [
        "Subject: win a free iPhone now! Click here to claim your prize.",
        "Subject: meeting reminder for tomorrow at 10am",
    ]
    for s in samples:
        print(f"{s[:60]}... -> {predict(s, model, vec)}")
