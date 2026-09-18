# Data Pipeline Documentation

## Purpose

This pipeline automates the complete data workflow from data collection to data validation.

## Pipeline Stages

### 1. Data Collection
- Script: `src/pipeline/collect.py`
- Input: Iris dataset from scikit-learn
- Output: `data/raw/iris_raw.csv`

### 2. Data Preprocessing
- Script: `src/pipeline/preprocess.py`
- Input: `data/raw/iris_raw.csv`
- Operations:
  - Remove duplicate rows
  - Convert numeric columns
  - Handle missing numeric values using median
  - Remove rows with missing species
  - Remove collection timestamp
- Output: `data/processed/iris_preprocessed.csv`

### 3. Feature Engineering
- Script: `src/pipeline/features.py`
- Input: `data/processed/iris_preprocessed.csv`
- Engineered features:
  - `sepal_area`
  - `petal_area`
  - `sepal_to_petal_length_ratio`
  - `petal_length_bin`
- Output: `data/processed/iris_features.csv`

### 4. Data Validation
- Script: `src/pipeline/validate.py`
- Input: `data/processed/iris_features.csv`
- Validation checks:
  - Required columns
  - Valid species values
  - Valid ranges for numeric features
- Output: Validation status

## DVC Pipeline

The pipeline is automated using `dvc.yaml`.

Pipeline flow:

collect
   ↓
preprocess
   ↓
features
   ↓
validate

DVC uses dependencies and output hashes to determine whether a stage needs to be re-executed.

## Verification

The pipeline was executed using:

dvc repro

A second execution showed that all stages were up to date and were skipped.

The pipeline graph was verified using:

dvc dag

The generated data was pushed to the configured DVC remote using:

dvc push

## Results

- Raw data collected: 150 rows
- Preprocessed data: 149 rows
- Engineered features: 9 columns
- Data validation: PASSED