import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay

from src.data import add_features, load_data, split_data
from src.features import load_vectorizer


def run_analysis():
    df = add_features(load_data())
    X_train, X_test, y_train, y_test = split_data(df)
    vec = load_vectorizer()
    X_test_vec = vec.transform(X_test)
    model = joblib.load("models/best_model.joblib")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ConfusionMatrixDisplay.from_estimator(model, X_test_vec, y_test, ax=axes[0])
    axes[0].set_title("Confusion Matrix")

    if hasattr(model, "predict_proba"):
        RocCurveDisplay.from_estimator(model, X_test_vec, y_test, ax=axes[1])
        PrecisionRecallDisplay.from_estimator(model, X_test_vec, y_test, ax=axes[2])
    else:
        RocCurveDisplay.from_estimator(model, X_test_vec, y_test, ax=axes[1])
        PrecisionRecallDisplay.from_estimator(model, X_test_vec, y_test, ax=axes[2])

    plt.tight_layout()
    plt.savefig("models/analysis.png", dpi=120)
    plt.show()

    y_pred = model.predict(X_test_vec)
    errors = X_test[(y_pred != y_test.values)]
    print(f"Ошибок: {len(errors)} из {len(X_test)}")
    print("\nПримеры ошибок:")
    for text in errors.head(5).values:
        print("—", text[:200], "\n")


if __name__ == "__main__":
    run_analysis()
