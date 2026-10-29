from mlflow.tracking import MlflowClient
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")

client = MlflowClient()
experiment = client.get_experiment_by_name("iris-hyperparameter-tuning")

if experiment is None:
    raise RuntimeError("Experiment 'iris-hyperparameter-tuning' was not found.")

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id]
)

print(f"{'Run Name':<32}{'CV F1 Macro':<16}{'Test Accuracy':<16}{'Total Fits':<12}")
print("-" * 76)

for run in sorted(
    runs,
    key=lambda r: r.data.tags.get("mlflow.runName", "")
):
    name = run.data.tags.get("mlflow.runName", "unknown")

    cv_score = run.data.metrics.get(
        "cv_f1_macro_mean",
        run.data.metrics.get("best_cv_f1_macro", 0.0)
    )
    test_accuracy = run.data.metrics.get("test_accuracy", 0.0)

    if name == "baseline_decision_tree":
        total_fits = 5
    elif name == "grid_search_random_forest":
        total_fits = int(run.data.params.get("total_combinations", 72)) * 5
    elif name == "random_search_random_forest":
        total_fits = int(run.data.params.get("n_iter", 30)) * 5
    else:
        total_fits = "-"

    print(
        f"{name:<32}{cv_score:<16.4f}"
        f"{test_accuracy:<16.4f}{str(total_fits):<12}"
    )
