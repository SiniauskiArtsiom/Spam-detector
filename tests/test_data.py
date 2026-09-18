import pytest
from src.data import clean_text


def test_clean_removes_punctuation():
    assert clean_text("Subject: Hello, world!") == "hello world"


def test_clean_lowercases():
    assert clean_text("Subject: HELLO WORLD") == "hello world"


def test_clean_removes_stopwords():
    result = clean_text("Subject: the cat is on the mat")
    assert "the" not in result.split()


def test_clean_handles_empty_after_subject():
    assert clean_text("Subject:") == ""
