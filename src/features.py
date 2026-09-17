import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

def build_tfidf(max_features: int = 5000) -> TfidfVectorizer:
    return TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))

def fit_transform(vectorizer, X_train, X_test):
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    return X_train_vec, X_test_vec

def save_vectorizer(vectorizer, path: str = "models/tfidf.joblib"):
    joblib.dump(vectorizer, path)

def load_vectorizer(path: str = "models/tfidf.joblib"):
    return joblib.load(path)
