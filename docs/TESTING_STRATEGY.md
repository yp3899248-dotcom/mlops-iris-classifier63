# Practical 8: Testing Strategy

## 1. Objective

The objective of this practical is to test data processing functions, feature engineering, data validation, the model training pipeline, and model quality using pytest.

## 2. Testing Frameworks

- **pytest:** Used to execute automated tests.
- **pytest-cov:** Used to measure code coverage.
- **JUnit XML:** Used to generate a machine-readable test results report.

## 3. Test Modules

### 3.1 Preprocessing Tests

File: `tests/test_preprocess.py`

Tests duplicate removal, missing numeric value handling, missing target removal, empty DataFrame handling, and command-line execution.

### 3.2 Feature Engineering Tests

File: `tests/test_features.py`

Tests the creation of engineered features, area calculations, length ratio calculation, zero petal-length handling, CSV output generation, and command-line execution.

### 3.3 Data Validation Tests

File: `tests/test_validate.py`

Tests valid input data, missing columns, unexpected species, out-of-range values, null values, and command-line execution with valid data.

### 3.4 Training Pipeline Tests

File: `tests/test_training_pipeline.py`

Tests model training using a synthetic dataset, prediction output shape, and prediction probability validity.

### 3.5 Model Quality Tests

File: `tests/test_model_quality.py`

Tests minimum model accuracy, valid species predictions, prediction count consistency, and prediction probability validity.

## 4. Test Execution Command

Run the following command to execute all tests and generate coverage and JUnit XML reports:

```bash
pytest --cov=src --cov-report=term-missing --junitxml=test-results.xml
```

## 5. Test Results

The combined test execution produced the following results:

- **Total tests:** 24
- **Passed:** 24
- **Failed:** 0
- **Overall source code coverage:** 25%

### Module Coverage

| Module | Coverage |
|---|---:|
| `src/pipeline/preprocess.py` | 100% |
| `src/pipeline/features.py` | 100% |
| `src/pipeline/validate.py` | 93% |
| `src/train.py` | 0% |

The overall coverage is 25% because several other project modules, including training and tuning scripts, are not covered by the current test suite. The three main data-processing modules have high coverage, with preprocessing and feature engineering reaching 100% and validation reaching 93%.

## 6. Test Report

The JUnit XML report was generated at:

`test-results.xml`

The report was verified to contain 24 test cases and a `<testsuite>` entry. It can be used by automated testing and continuous integration tools.

## 7. Conclusion

Automated tests were created for preprocessing, feature engineering, data validation, model training smoke tests, and model quality. All 24 tests passed successfully. The preprocessing and feature engineering modules achieved 100% coverage, while the validation module achieved 93% coverage. The overall source code coverage is 25% because several project modules remain untested. These tests help identify errors early and improve the reliability and maintainability of the Iris classification project.