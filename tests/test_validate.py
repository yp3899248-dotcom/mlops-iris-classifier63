import pandas as pd
import pytest

from src.pipeline.validate import validate, DataValidationError


def create_valid_data():
    return pd.DataFrame({
        "sepal length (cm)": [5.1, 6.0],
        "sepal width (cm)": [3.5, 2.8],
        "petal length (cm)": [1.4, 4.5],
        "petal width (cm)": [0.2, 1.5],
        "species": ["setosa", "versicolor"],
        "sepal_area": [17.85, 16.8],
        "petal_area": [0.28, 6.75],
        "sepal_to_petal_length_ratio": [3.64, 1.33],
        "petal_length_bin": ["short", "medium"],
    })


def test_validate_accepts_valid_data(tmp_path):
    input_path = tmp_path / "valid.csv"
    create_valid_data().to_csv(input_path, index=False)

    result = validate(str(input_path))

    assert len(result) == 2


def test_validate_rejects_missing_column(tmp_path):
    input_path = tmp_path / "missing.csv"
    df = create_valid_data().drop(columns=["sepal_area"])
    df.to_csv(input_path, index=False)

    with pytest.raises(DataValidationError):
        validate(str(input_path))


def test_validate_rejects_unexpected_species(tmp_path):
    input_path = tmp_path / "species.csv"
    df = create_valid_data()
    df.loc[0, "species"] = "unknown_species"
    df.to_csv(input_path, index=False)

    with pytest.raises(DataValidationError):
        validate(str(input_path))


def test_validate_rejects_out_of_range_values(tmp_path):
    input_path = tmp_path / "range.csv"
    df = create_valid_data()
    df.loc[0, "sepal length (cm)"] = 10.0
    df.to_csv(input_path, index=False)

    with pytest.raises(DataValidationError):
        validate(str(input_path))


def test_validate_rejects_null_values(tmp_path):
    input_path = tmp_path / "null.csv"
    df = create_valid_data()
    df.loc[0, "species"] = None
    df.to_csv(input_path, index=False)

    with pytest.raises(DataValidationError):
        validate(str(input_path))
def test_validate_cli_runs_successfully(tmp_path, monkeypatch):
    import runpy
    import sys
    from pathlib import Path

    input_file = tmp_path / "valid_input.csv"

    pd.DataFrame(
        {
            "sepal length (cm)": [5.1, 6.0],
            "sepal width (cm)": [3.5, 2.8],
            "petal length (cm)": [1.4, 4.5],
            "petal width (cm)": [0.2, 1.5],
            "species": ["setosa", "versicolor"],
            "sepal_area": [17.85, 16.8],
            "petal_area": [0.28, 6.75],
            "sepal_to_petal_length_ratio": [3.64, 1.33],
            "petal_length_bin": ["short", "long"],
        }
    ).to_csv(input_file, index=False)

    script_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "pipeline"
        / "validate.py"
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [str(script_path), "--input", str(input_file)],
    )

    runpy.run_path(str(script_path), run_name="__main__")