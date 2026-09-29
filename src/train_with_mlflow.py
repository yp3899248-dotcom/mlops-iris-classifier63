import os
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

DATA_PATH = "data/processed/iris_features.csv"

FEATURES = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
    "sepal_area",
    "petal_area",
    "sepal_to_petal_length_ratio",
]

TARGET = "species"

EXPERIMENT_NAME = "iris-classification-baseline"

mlflow.set_experiment(EXPERIMENT_NAME)

df = pd.read_csv(DATA_PATH)

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

models = [
    (
        "logistic_regression",
        LogisticRegression(max_iter=200, C=1.0),
        {
            "model_type": "LogisticRegression",
            "max_iter": 200,
            "C": 1.0,
        },
    ),
    (
        "random_forest_shallow",
        RandomForestClassifier(
            n_estimators=50,
            max_depth=3,
            random_state=42
        ),
        {
            "model_type": "RandomForest",
            "n_estimators": 50,
            "max_depth": 3,
        },
    ),
    (
        "random_forest_deep",
        RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=42
        ),
        {
            "model_type": "RandomForest",
            "n_estimators": 200,
            "max_depth": "None",
        },
    ),
]

results = []

for run_name, model, params in models:

    with mlflow.start_run(run_name=run_name):

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(
            y_test, y_pred, average="macro"
        )
        recall = recall_score(
            y_test, y_pred, average="macro"
        )
        f1 = f1_score(
            y_test, y_pred, average="macro"
        )

        mlflow.log_params(params)

        mlflow.log_metrics({
            "accuracy": accuracy,
            "precision_macro": precision,
            "recall_macro": recall,
            "f1_macro": f1,
        })

        cm = confusion_matrix(y_test, y_pred)

        cm_path = f"confusion_matrix_{run_name}.csv"
        pd.DataFrame(cm).to_csv(
            cm_path,
            index=False
        )

        mlflow.log_artifact(cm_path)

        mlflow.sklearn.log_model(
            model,
            artifact_path="model"
        )

        results.append({
            "run_name": run_name,
            "accuracy": accuracy,
            "precision_macro": precision,
            "recall_macro": recall,
            "f1_macro": f1,
        })

        print(
            f"{run_name}: "
            f"accuracy={accuracy:.4f}, "
            f"f1_macro={f1:.4f}"
        )

results_df = pd.DataFrame(results)

print("\nResults:")
print(results_df)

best_run = results_df.loc[
    results_df["f1_macro"].idxmax()
]

print("\nBest model:")
print(best_run)