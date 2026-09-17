from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

def get_models():
    return {
        "LogReg": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                                random_state=42, n_jobs=-1),
        "LinearSVC": LinearSVC(class_weight='balanced', random_state=42, max_iter=5000),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                  use_label_encoder=False, eval_metric='logloss',
                                  random_state=42, n_jobs=-1),
    }

def get_param_grids():
    return {
        "LogReg": {"C": [0.1, 1.0, 10.0]},
        "RandomForest": {"n_estimators": [100, 200], "max_depth": [None, 20, 40]},
        "LinearSVC": {"C": [0.1, 1.0, 10.0]},
        "XGBoost": {"n_estimators": [200, 300], "max_depth": [4, 6, 8]},
    }
