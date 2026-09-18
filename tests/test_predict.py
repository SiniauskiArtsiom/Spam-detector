from pathlib import Path
import pytest
from src.predict import predict, load_artifacts

MODEL_PATH = Path("models/best_model.joblib")
VEC_PATH = Path("models/tfidf.joblib")


@pytest.mark.skipif(
    not (MODEL_PATH.exists() and VEC_PATH.exists()),
    reason="model not trained yet"
)
def test_predict_returns_valid_output():
    model, vec = load_artifacts()
    result = predict("Subject: win a free iPhone now!", model, vec)
    assert result["label"] in (0, 1)
    assert result["probability"] is not None
    assert 0.0 <= result["probability"] <= 1.0
