import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


FEATURE_COLUMNS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]


@pytest.fixture
def trained_model_and_test_data():
    data = pd.read_csv("data/processed/iris_features.csv")

    X = data[FEATURE_COLUMNS]
    y = data["species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )
    model.fit(X_train, y_train)

    return model, X_test, y_test, y.unique()


def test_model_accuracy_meets_minimum_threshold(
    trained_model_and_test_data,
):
    model, X_test, y_test, _ = trained_model_and_test_data

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    assert accuracy >= 0.90, (
        f"Model accuracy {accuracy:.2%} is below the required 90%"
    )


def test_model_predicts_valid_species(trained_model_and_test_data):
    model, X_test, _, valid_species = trained_model_and_test_data

    predictions = model.predict(X_test)

    assert set(predictions).issubset(set(valid_species))


def test_model_prediction_count_matches_test_data(
    trained_model_and_test_data,
):
    model, X_test, y_test, _ = trained_model_and_test_data

    predictions = model.predict(X_test)

    assert len(predictions) == len(y_test)


def test_model_prediction_probabilities_are_valid(
    trained_model_and_test_data,
):
    model, X_test, _, _ = trained_model_and_test_data

    probabilities = model.predict_proba(X_test)

    assert probabilities.shape[0] == len(X_test)
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)