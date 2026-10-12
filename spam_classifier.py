"""Educational Multinomial Naive Bayes email classifier (standard library only)."""
from __future__ import annotations

import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

TOKEN_RE = re.compile(r"[a-z0-9']+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    """Lowercase and extract simple English word tokens."""
    return TOKEN_RE.findall(text.lower())


class NaiveBayesSpamClassifier:
    """Multinomial NB with Laplace smoothing and log probabilities."""

    def __init__(self, alpha: float = 1.0):
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.alpha = alpha
        self.labels = ("ham", "spam")
        self.document_counts = Counter()
        self.word_counts = {label: Counter() for label in self.labels}
        self.total_words = Counter()
        self.vocabulary: set[str] = set()
        self.fitted = False

    def fit(self, examples: list[tuple[str, str]]) -> "NaiveBayesSpamClassifier":
        if not examples:
            raise ValueError("Training examples are required")
        self.document_counts.clear()
        self.word_counts = {label: Counter() for label in self.labels}
        self.total_words.clear()
        self.vocabulary.clear()
        for label, text in examples:
            if label not in self.labels:
                raise ValueError(f"Unknown label: {label}")
            tokens = tokenize(text)
            self.document_counts[label] += 1
            self.word_counts[label].update(tokens)
            self.total_words[label] += len(tokens)
            self.vocabulary.update(tokens)
        if any(self.document_counts[label] == 0 for label in self.labels):
            raise ValueError("Both spam and ham examples are required")
        self.fitted = True
        return self

    def predict_proba(self, text: str) -> dict[str, float]:
        if not self.fitted:
            raise RuntimeError("Call fit() first")
        doc_total = sum(self.document_counts.values())
        vocab_size = max(1, len(self.vocabulary))
        scores = {}
        for label in self.labels:
            log_score = math.log(self.document_counts[label] / doc_total)
            denominator = self.total_words[label] + self.alpha * vocab_size
            for token, frequency in Counter(tokenize(text)).items():
                numerator = self.word_counts[label][token] + self.alpha
                log_score += frequency * math.log(numerator / denominator)
            scores[label] = log_score
        max_log = max(scores.values())
        weights = {label: math.exp(score - max_log) for label, score in scores.items()}
        total = sum(weights.values())
        return {label: weights[label] / total for label in self.labels}

    def predict(self, text: str) -> str:
        probabilities = self.predict_proba(text)
        return max(probabilities, key=probabilities.get)


def load_examples(path: str | Path) -> list[tuple[str, str]]:
    with open(path, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if not {"label", "text"}.issubset(reader.fieldnames or []):
            raise ValueError("CSV must have label,text columns")
        return [(row["label"].strip().lower(), row["text"].strip()) for row in reader]


def split_examples(examples: list[tuple[str, str]], test_fraction: float = 0.25, seed: int = 42):
    """Deterministic stratified split with both labels represented in each split."""
    import random
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    groups = defaultdict(list)
    for label, text in examples:
        groups[label].append((label, text))
    train, test = [], []
    randomizer = random.Random(seed)
    for label in ("ham", "spam"):
        group = groups[label][:]
        if len(group) < 2:
            raise ValueError(f"At least two {label} examples required")
        randomizer.shuffle(group)
        n_test = max(1, min(len(group) - 1, round(len(group) * test_fraction)))
        test.extend(group[:n_test])
        train.extend(group[n_test:])
    return train, test


def evaluate(model, examples: list[tuple[str, str]]) -> dict[str, float | int]:
    tp = fp = tn = fn = 0
    for actual, text in examples:
        prediction = model.predict(text)
        if actual == "spam" and prediction == "spam": tp += 1
        elif actual == "ham" and prediction == "spam": fp += 1
        elif actual == "ham" and prediction == "ham": tn += 1
        else: fn += 1
    n = len(examples)
    return {
        "accuracy": (tp + tn) / n if n else 0.0,
        "precision_spam": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall_spam": tp / (tp + fn) if (tp + fn) else 0.0,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }
