import pandas as pd

from src.pipeline.features import engineer_features


def create_input_csv(tmp_path, data):
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "features.csv"

    pd.DataFrame(data).to_csv(input_path, index=False)

    return input_path, output_path


def test_engineer_features_creates_expected_columns(tmp_path):
    data = {
        "sepal length (cm)": [5.0, 6.0],
        "sepal width (cm)": [3.0, 2.5],
        "petal length (cm)": [1.5, 4.0],
        "petal width (cm)": [0.2, 1.2],
        "species": ["setosa", "versicolor"],
    }

    input_path, output_path = create_input_csv(tmp_path, data)

    result = engineer_features(str(input_path), str(output_path))

    expected_columns = [
        "sepal_area",
        "petal_area",
        "sepal_to_petal_length_ratio",
        "petal_length_bin",
    ]

    for column in expected_columns:
        assert column in result.columns

    assert output_path.exists()


def test_engineer_features_calculates_areas_correctly(tmp_path):
    data = {
        "sepal length (cm)": [5.0],
        "sepal width (cm)": [3.0],
        "petal length (cm)": [2.0],
        "petal width (cm)": [0.5],
        "species": ["setosa"],
    }

    input_path, output_path = create_input_csv(tmp_path, data)

    result = engineer_features(str(input_path), str(output_path))

    assert result["sepal_area"].iloc[0] == 15.0
    assert result["petal_area"].iloc[0] == 1.0


def test_engineer_features_calculates_length_ratio(tmp_path):
    data = {
        "sepal length (cm)": [6.0],
        "sepal width (cm)": [3.0],
        "petal length (cm)": [2.0],
        "petal width (cm)": [0.5],
        "species": ["versicolor"],
    }

    input_path, output_path = create_input_csv(tmp_path, data)

    result = engineer_features(str(input_path), str(output_path))

    assert result["sepal_to_petal_length_ratio"].iloc[0] == 3.0


def test_engineer_features_handles_zero_petal_length(tmp_path):
    data = {
        "sepal length (cm)": [5.0],
        "sepal width (cm)": [3.0],
        "petal length (cm)": [0.0],
        "petal width (cm)": [0.0],
        "species": ["setosa"],
    }

    input_path, output_path = create_input_csv(tmp_path, data)

    result = engineer_features(str(input_path), str(output_path))

    ratio = result["sepal_to_petal_length_ratio"].iloc[0]

    assert pd.isna(ratio) or ratio == float("inf") or ratio == float("-inf")


def test_engineer_features_saves_output_csv(tmp_path):
    data = {
        "sepal length (cm)": [5.0],
        "sepal width (cm)": [3.0],
        "petal length (cm)": [1.5],
        "petal width (cm)": [0.2],
        "species": ["setosa"],
    }

    input_path, output_path = create_input_csv(tmp_path, data)

    engineer_features(str(input_path), str(output_path))

    assert output_path.exists()

    saved_result = pd.read_csv(output_path)

    assert "sepal_area" in saved_result.columns
    assert "petal_area" in saved_result.columns