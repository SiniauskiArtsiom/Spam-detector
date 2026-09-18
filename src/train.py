import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV

from src.data import add_features, load_data, split_data
from src.evaluate import get_metrics
from src.features import build_tfidf, fit_transform, save_vectorizer
from src.models import get_models, get_param_grids


def train(tune: bool = True):
    df = add_features(load_data())
    X_train, X_test, y_train, y_test = split_data(df)

    vectorizer = build_tfidf()
    X_train_vec, X_test_vec = fit_transform(vectorizer, X_train, X_test)
    save_vectorizer(vectorizer)

    models = get_models()
    grids = get_param_grids()
    results = {}
    best_model, best_f1, best_name = None, 0, ""

    for name, model in models.items():
        if tune:
            search = GridSearchCV(model, grids[name], cv=3, scoring="f1", n_jobs=-1, verbose=0)
            search.fit(X_train_vec, y_train)
            model = search.best_estimator_
            print(f"{name} best params: {search.best_params_}")
        else:
            model.fit(X_train_vec, y_train)

        metrics = get_metrics(model, X_test_vec, y_test)
        results[name] = metrics
        print(f"{name}: {metrics}")

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_model = model
            best_name = name

    print(f"\nЛучшая модель: {best_name} (F1={best_f1:.4f})")
    joblib.dump(best_model, "models/best_model.joblib")
    joblib.dump(best_name, "models/best_model_name.joblib")

    pd.DataFrame(results).T.to_csv("models/results.csv")
    print("\nРезультаты сохранены в models/results.csv")
    return best_model, results


if __name__ == "__main__":
    train()
