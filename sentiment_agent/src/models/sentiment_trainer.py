"""Train a TF-IDF + LogisticRegression sentiment classifier for Indian financial news."""

import csv
import joblib
import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


def _clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s%.,;]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def train(csv_path: str | Path, model_path: str | Path, test_size: float = 0.2):
    """Train and save sentiment classifier."""
    csv_path = Path(csv_path)
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # Load
    texts, labels = [], []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(_clean(row["text"]))
            labels.append(row["label"])

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=42, stratify=labels
    )

    # Pipeline: TF-IDF + LogisticRegression
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            C=1.0,
            random_state=42,
        )),
    ])

    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {acc:.3f}")
    print("Note: the training CSV is template-generated, so this held-out accuracy is optimistic;\n"
          "      it does not measure accuracy on real headlines.")
    print(classification_report(y_test, y_pred, digits=3))

    # Save
    joblib.dump(pipeline, model_path)
    print(f"Model saved -> {model_path}")
    return model_path, acc


if __name__ == "__main__":
    from src.config.settings import LOCAL_DATA
    train(
        LOCAL_DATA / "sentiment_training.csv",
        LOCAL_DATA / "models" / "sentiment_clf.joblib",
    )
