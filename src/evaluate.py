from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def get_metrics(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    metrics = {
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
        metrics["pr_auc"] = average_precision_score(y_test, y_proba)
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
        metrics["roc_auc"] = roc_auc_score(y_test, y_score)
        metrics["pr_auc"] = average_precision_score(y_test, y_score)
    return metrics


def print_metrics(model, X_test, y_test, name: str = "Model"):
    print(f"\n=== {name} ===")
    print(classification_report(y_test, model.predict(X_test)))
    m = get_metrics(model, X_test, y_test)
    print({k: round(v, 4) for k, v in m.items()})
    return m


def print_confusion(model, X_test, y_test):
    print(confusion_matrix(y_test, model.predict(X_test)))
