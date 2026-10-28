\# Practical 8: Testing Strategy



\## 1. Objective



The objective of this practical is to test data processing functions, feature engineering, data validation, the model training pipeline, and model quality using pytest.



\## 2. Testing Frameworks



\- \*\*pytest:\*\* Used to execute automated tests.

\- \*\*pytest-cov:\*\* Used to measure code coverage.

\- \*\*JUnit XML:\*\* Used to generate a machine-readable test results report.



\## 3. Test Modules



\### 3.1 Preprocessing Tests



File: `tests/test\_preprocess.py`



Tests duplicate removal, missing numeric value handling, missing target removal, and empty DataFrame handling.



\### 3.2 Feature Engineering Tests



File: `tests/test\_features.py`



Tests the creation of engineered features, area calculations, length ratio calculation, zero petal-length handling, and CSV output generation.



\### 3.3 Data Validation Tests



File: `tests/test\_validate.py`



Tests valid input data, missing columns, unexpected species, out-of-range values, and null values.



\### 3.4 Training Pipeline Tests



File: `tests/test\_training\_pipeline.py`



Tests data splitting, feature count, Random Forest model training, and model evaluation output.



\### 3.5 Model Quality Tests



File: `tests/test\_model\_quality.py`



Tests minimum model accuracy, valid species predictions, and prediction count consistency.



\## 4. Test Execution Command



Run the following command to execute all tests and generate coverage and JUnit XML reports:



```bash

pytest --cov=src --cov-report=term-missing --junitxml=test-results.xml

```



\## 5. Test Results



The combined test execution produced the following results:



\- Total tests: 21

\- Passed: 21

\- Failed: 0

\- Overall source code coverage: 26%



\### Module Coverage



| Module | Coverage |

|---|---:|

| `src/pipeline/preprocess.py` | 83% |

| `src/pipeline/features.py` | 75% |

| `src/pipeline/validate.py` | 81% |

| `src/train.py` | 70% |



The overall coverage is 26% because several other project modules are not covered by the current test suite.



\## 6. Test Report



The JUnit XML report was generated at:



`test-results.xml`



This report can be used by automated testing and continuous integration tools.



\## 7. Conclusion



Automated tests were created for preprocessing, feature engineering, data validation, model training, and model quality. All 21 tests passed successfully. The tests help identify errors early and improve the reliability and maintainability of the Iris classification project.

