"""Classifiers on fixed features, and evaluation metrics (numpy + scikit-learn)."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score


def l2_normalise(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def full_proba(model, X: np.ndarray, n_classes: int) -> np.ndarray:
    """predict_proba with a column for every class, even if the model saw only some of them."""
    p = model.predict_proba(X)
    out = np.zeros((len(X), n_classes))
    out[:, model.classes_.astype(int)] = p
    return out


def linear_probe(train_f: np.ndarray, train_y: np.ndarray, test_f: np.ndarray, n_classes: int,
                 C: float = 1.0, max_iter: int = 5000) -> np.ndarray:
    """L2-normalised features -> multinomial logistic regression with a fixed regularisation strength
    (no validation split exists at 1-5 shots, so nothing is tuned)."""
    clf = LogisticRegression(C=C, max_iter=max_iter)
    clf.fit(l2_normalise(train_f), train_y)
    return full_proba(clf, l2_normalise(test_f), n_classes)


def random_forest(train_f: np.ndarray, train_y: np.ndarray, test_f: np.ndarray, n_classes: int,
                  n_estimators: int = 500, seed: int = 0) -> np.ndarray:
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=-1)
    clf.fit(train_f, train_y)
    return full_proba(clf, test_f, n_classes)


def scores(y_true: np.ndarray, probs: np.ndarray) -> dict:
    """Balanced accuracy (primary: robust to class imbalance), accuracy, macro one-vs-rest AUC."""
    pred = probs.argmax(1)
    n_classes = probs.shape[1]
    try:
        if n_classes == 2:
            auc = roc_auc_score(y_true, probs[:, 1])
        else:
            p = probs / np.clip(probs.sum(1, keepdims=True), 1e-12, None)
            auc = roc_auc_score(y_true, p, multi_class="ovr", average="macro", labels=np.arange(n_classes))
    except ValueError:
        auc = float("nan")
    return {"balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
            "accuracy": float(accuracy_score(y_true, pred)), "auc": float(auc)}
