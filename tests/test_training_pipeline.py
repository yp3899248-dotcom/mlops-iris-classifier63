import numpy as np
import pytest
import joblib

from sklearn.ensemble import RandomForestClassifier

from src.train import load_data, train_model, evaluate_model


def test_load_data_returns_expected_shapes():
    X_train, X_test, y_train, y_test = load_data()

    assert X_train.shape == (120, 4)
    assert X_test.shape == (30, 4)
    assert len(y_train) == 120
    assert len(y_test) == 30


def test_load_data_has_no_missing_values():
    X_train, X_test, y_train, y_test = load_data()

    assert not np.isnan(X_train).any()
    assert not np.isnan(X_test).any()
    assert not np.isnan(y_train).any()
    assert not np.isnan(y_test).any()


def test_train_model_returns_random_forest():
    X_train, X_test, y_train, y_test = load_data()

    model = train_model(X_train, y_train, n_estimators=10)

    assert isinstance(model, RandomForestClassifier)
    assert model.n_estimators == 10
    assert hasattr(model, "predict")


def test_train_model_predicts_correct_number_of_rows():
    X_train, X_test, y_train, y_test = load_data()

    model = train_model(X_train, y_train, n_estimators=10)
    predictions = model.predict(X_test)

    assert len(predictions) == len(y_test)


def test_evaluate_model_returns_accuracy_and_report():
    X_train, X_test, y_train, y_test = load_data()

    model = train_model(X_train, y_train, n_estimators=10)
    accuracy, report = evaluate_model(model, X_test, y_test)

    assert 0.0 <= accuracy <= 1.0
    assert isinstance(report, str)
    assert "precision" in report
    assert "recall" in report


def test_evaluate_model_accuracy_meets_minimum():
    X_train, X_test, y_train, y_test = load_data()

    model = train_model(X_train, y_train)
    accuracy, report = evaluate_model(model, X_test, y_test)

    assert accuracy >= 0.90


def test_main_trains_and_saves_model(tmp_path, monkeypatch, capsys):
    import src.train as train_module

    def fake_load_data():
        X_train = np.array(
            [
                [1.0, 2.0, 1.0, 0.1],
                [2.0, 3.0, 2.0, 0.2],
                [3.0, 4.0, 3.0, 0.3],
                [4.0, 5.0, 4.0, 0.4],
                [5.0, 6.0, 5.0, 0.5],
                [6.0, 7.0, 6.0, 0.6],
            ]
        )
        X_test = np.array([[1.5, 2.5, 1.5, 0.15]])
        y_train = np.array([0, 0, 1, 1, 2, 2])
        y_test = np.array([0])
        return X_train, X_test, y_train, y_test

    monkeypatch.setattr(train_module, "load_data", fake_load_data)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models").mkdir()

    train_module.main()

    output = capsys.readouterr().out
    assert "Accuracy:" in output
    assert "Model saved to models/iris_model.joblib" in output

    model_path = tmp_path / "models" / "iris_model.joblib"
    assert model_path.exists()

    saved_model = joblib.load(model_path)
    assert isinstance(saved_model, RandomForestClassifier)