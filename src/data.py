import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

LEMMATIZER = WordNetLemmatizer()
STOP_WORDS = set(stopwords.words('english'))
FIRST_N_SKIP = 9  # "Subject: "

def load_data(path: str = "data/raw/emails.csv") -> pd.DataFrame:
    return pd.read_csv(path)

def clean_text(text: str) -> str:
    text = text[FIRST_N_SKIP:].lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    tokens = text.split()
    clean_tokens = [LEMMATIZER.lemmatize(w) for w in tokens if w not in STOP_WORDS]
    return " ".join(clean_tokens)

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['clean_text'] = df['text'].apply(clean_text)
    df['length'] = df['text'].apply(len)
    df['punct_percent'] = df['text'].apply(
        lambda t: sum(ch in string.punctuation for ch in t) / max(len(t), 1) * 100
    )
    df['caps_ratio'] = df['text'].apply(
        lambda t: sum(ch.isupper() for ch in t) / max(len(t), 1) * 100
    )
    df['digit_percent'] = df['text'].apply(
        lambda t: sum(ch.isdigit() for ch in t) / max(len(t), 1) * 100
    )
    return df

def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    return train_test_split(
        df['clean_text'], df['spam'],
        test_size=test_size, random_state=random_state, stratify=df['spam']
    )

if __name__ == "__main__":
    df = add_features(load_data())
    print(df[['text', 'clean_text', 'spam']].head())
    print(f"Размер: {df.shape}")
    print(f"Дисбаланс: {df['spam'].value_counts(normalize=True).to_dict()}")
