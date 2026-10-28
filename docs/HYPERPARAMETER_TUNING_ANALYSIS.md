# Practical 7: Hyperparameter Tuning and Comparative Performance Analysis

## 1. Objective

The objective of this practical is to establish a baseline model, perform hyperparameter tuning using Grid Search and Random Search, track experiments using MLflow, and compare the performance of the models.

## 2. Dataset and Evaluation

- Dataset: Iris dataset with engineered features.
- Baseline model: Decision Tree Classifier.
- Tuned model: Random Forest Classifier.
- Evaluation metric: Macro F1-score using 5-fold cross-validation.
- Additional metric: Test accuracy.
- Experiment tracking: MLflow.
- MLflow experiment name: `iris-hyperparameter-tuning`.

## 3. Experimental Results

| Model | CV F1 Macro | Test Accuracy | Total Fits |
|---|---:|---:|---:|
| Baseline Decision Tree | 0.9663 | 0.9000 | 5 |
| Grid Search Random Forest | 0.9663 | 0.9667 | 360 |
| Random Search Random Forest | 0.9663 | 0.9667 | 150 |

## 4. Baseline Model

A Decision Tree Classifier with default hyperparameters was used as the baseline model.

Results:
- Mean CV F1 Macro: 0.9663
- Standard deviation of CV F1 Macro: 0.0316
- Test accuracy: 90.00%

The baseline provides a reference for evaluating the tuned Random Forest models.

## 5. Grid Search

GridSearchCV was used to evaluate 72 hyperparameter combinations with 5-fold cross-validation, resulting in 360 model fits.

Best hyperparameters:

- `n_estimators`: 50
- `max_depth`: 3
- `min_samples_split`: 2
- `max_features`: sqrt

Results:
- Best CV F1 Macro: 0.9663
- Test accuracy: 96.67%

The complete candidate results were saved in `grid_search_all_candidates.csv` and logged as an MLflow artifact.

## 6. Random Search

RandomizedSearchCV was used to evaluate 30 sampled hyperparameter combinations with 5-fold cross-validation, resulting in 150 model fits.

Best hyperparameters:

- `n_estimators`: 100
- `max_depth`: 3
- `min_samples_split`: 6
- `max_features`: sqrt

Results:
- Best CV F1 Macro: 0.9663
- Test accuracy: 96.67%

The complete candidate results were saved in `random_search_all_candidates.csv` and logged as an MLflow artifact.

## 7. Comparative Analysis

The baseline Decision Tree achieved 90.00% test accuracy. Both tuned Random Forest models achieved 96.67% test accuracy, an improvement of approximately 6.67 percentage points over the baseline.

All three models achieved the same reported mean CV F1 Macro score of 0.9663. Therefore, hyperparameter tuning did not improve the reported cross-validation F1 score in this experiment.

Grid Search evaluated 360 fits, whereas Random Search evaluated 150 fits. Random Search used approximately 58.33% fewer fits than Grid Search while achieving the same test accuracy in this run.

These results suggest that Random Search was more computationally efficient for this experiment. However, this conclusion is based on the observed results and does not guarantee that Random Search will always outperform Grid Search.

## 8. MLflow Tracking

The experiments were tracked under the MLflow experiment `iris-hyperparameter-tuning`.

The logged information includes:
- Model and search parameters.
- Cross-validation F1 Macro scores.
- Test accuracy.
- Best hyperparameters for the tuned models.
- Candidate-search result CSV files as artifacts.

The experiments can be inspected using the MLflow UI.

## 9. Conclusion

This practical demonstrated baseline model development, hyperparameter tuning with GridSearchCV and RandomizedSearchCV, experiment tracking with MLflow, and comparative performance analysis.

Both tuned Random Forest models achieved 96.67% test accuracy compared with 90.00% for the baseline Decision Tree. Their reported CV F1 Macro scores were identical at 0.9663.

Random Search reached the same test accuracy as Grid Search using 150 fits instead of 360. Thus, Random Search provided a more computationally efficient result for this particular experiment.