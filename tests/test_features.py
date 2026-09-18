from src.features import build_tfidf


def test_tfidf_shape():
    vec = build_tfidf(max_features=100)
    X = vec.fit_transform(["hello world", "spam spam spam"])
    assert X.shape[0] == 2
    assert X.shape[1] <= 100
