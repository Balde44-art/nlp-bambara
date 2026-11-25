"""Détecteur de langue : n-grammes de caractères + TF-IDF + régression logistique."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "langid.joblib"


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 4), lowercase=True),
            ),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )


def train(texts: Sequence[str], labels: Sequence[str]) -> Pipeline:
    model = build_pipeline()
    model.fit(list(texts), list(labels))
    return model


def predict(model: Pipeline, texts: Sequence[str]) -> list[str]:
    return list(model.predict(list(texts)))


def predict_proba(model: Pipeline, text: str) -> dict[str, float]:
    probs = model.predict_proba([text])[0]
    return dict(zip(model.classes_, (float(p) for p in probs)))


def save(model: Pipeline, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load(path: Path = MODEL_PATH) -> Pipeline:
    return joblib.load(path)
