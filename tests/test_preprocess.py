import pandas as pd

from src.pipeline.preprocess import preprocess


def create_test_csv(tmp_path, data):
    """Create a temporary CSV file for testing."""
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"

    df = pd.DataFrame(data)
    df.to_csv(input_path, index=False)

    return input_path, output_path


def test_preprocess_removes_duplicates(tmp_path):
    """Test whether duplicate rows are removed."""

    data = {
        "sepal length (cm)": [5.1, 5.1, 6.2],
        "sepal width (cm)": [3.5, 3.5, 3.4],
        "petal length (cm)": [1.4, 1.4, 4.5],
        "petal width (cm)": [0.2, 0.2, 1.5],
        "species": ["setosa", "setosa", "versicolor"],
    }

    input_path, output_path = create_test_csv(tmp_path, data)

    result = preprocess(str(input_path), str(output_path))

    assert len(result) == 2
    assert not result.duplicated().any()
    assert output_path.exists()


def test_preprocess_handles_missing_numeric_values(tmp_path):
    """Test whether missing numeric values are filled with the median."""

    data = {
        "sepal length (cm)": [5.1, None, 6.2],
        "sepal width (cm)": [3.5, 3.0, 3.4],
        "petal length (cm)": [1.4, 1.5, 4.5],
        "petal width (cm)": [0.2, 0.3, 1.5],
        "species": ["setosa", "setosa", "versicolor"],
    }

    input_path, output_path = create_test_csv(tmp_path, data)

    result = preprocess(str(input_path), str(output_path))

    assert result["sepal length (cm)"].isna().sum() == 0
    assert result["sepal length (cm)"].iloc[1] == 5.65


def test_preprocess_drops_missing_target(tmp_path):
    """Test whether rows with missing species are removed."""

    data = {
        "sepal length (cm)": [5.1, 6.2, 5.8],
        "sepal width (cm)": [3.5, 3.4, 2.7],
        "petal length (cm)": [1.4, 4.5, 5.1],
        "petal width (cm)": [0.2, 1.5, 1.9],
        "species": ["setosa", None, "virginica"],
    }

    input_path, output_path = create_test_csv(tmp_path, data)

    result = preprocess(str(input_path), str(output_path))

    assert result["species"].isna().sum() == 0
    assert len(result) == 2


def test_preprocess_handles_empty_dataframe(tmp_path):
    """Test preprocessing an empty CSV with the required columns."""

    data = {
        "sepal length (cm)": [],
        "sepal width (cm)": [],
        "petal length (cm)": [],
        "petal width (cm)": [],
        "species": [],
    }

    input_path, output_path = create_test_csv(tmp_path, data)

    result = preprocess(str(input_path), str(output_path))

    assert isinstance(result, pd.DataFrame)
    assert result.empty
    assert output_path.exists()