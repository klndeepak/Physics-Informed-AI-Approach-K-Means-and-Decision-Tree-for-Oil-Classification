"""Tests for raman_analysis.metrics."""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from raman_analysis.metrics import classification_performance, plot_confusion_matrix


def _toy_classifier():
    X = pd.DataFrame({"f1": [0, 0, 1, 1], "f2": [0, 1, 0, 1]})
    y = pd.Series(["a", "a", "b", "b"])
    model = DecisionTreeClassifier(random_state=0).fit(X, y)
    return model, X, y


def test_classification_performance_reports_perfect_fit():
    model, X, y = _toy_classifier()

    performance = classification_performance(model, X, y)

    assert list(performance.columns) == ["Accuracy", "Recall", "Precision", "F1"]
    assert (performance.iloc[0] == 1.0).all()


def test_plot_confusion_matrix_writes_a_file(tmp_path):
    model, X, y = _toy_classifier()
    out_path = tmp_path / "confusion.jpg"

    plot_confusion_matrix(model, X, y, class_labels=["a", "b"], title="Toy", out_path=out_path)

    assert out_path.exists()
    assert out_path.stat().st_size > 0
