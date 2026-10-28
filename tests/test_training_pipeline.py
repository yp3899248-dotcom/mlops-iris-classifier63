import numpy as np

from sklearn.ensemble import RandomForestClassifier

from src.train import load_data, train_model, evaluate_model


def test_load_data_returns_train_test_splits():
    X_train, X_test, y_train, y_test = load_data()

    assert len(X_train) == 120
    assert len(X_test) == 30
    assert len(y_train) == 120
    assert len(y_test) == 30


def test_load_data_has_expected_feature_count():
    X_train, X_test, y_train, y_test = load_data()

    assert X_train.shape[1] == 4
    assert X_test.shape[1] == 4


def test_train_model_returns_trained_random_forest():
    X_train, X_test, y_train, y_test = load_data()

    model = train_model(X_train, y_train)

    assert isinstance(model, RandomForestClassifier)
    assert hasattr(model, "predict")
    assert len(model.classes_) == 3


def test_evaluate_model_returns_accuracy_and_report():
    X_train, X_test, y_train, y_test = load_data()

    model = train_model(X_train, y_train)

    accuracy, report = evaluate_model(model, X_test, y_test)

    assert 0.0 <= accuracy <= 1.0
    assert isinstance(report, str)
    assert "precision" in report
    assert "recall" in report