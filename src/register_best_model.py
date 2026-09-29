import mlflow
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "iris-classification-baseline"
MODEL_NAME = "iris-classifier-prod"

client = MlflowClient()

experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.f1_macro DESC"],
    max_results=1,
)

best_run = runs[0]

run_id = best_run.info.run_id
f1_score = best_run.data.metrics["f1_macro"]

print("Best run:", run_id)
print("Best F1:", f1_score)

model_uri = f"runs:/{run_id}/model"

try:
    client.get_registered_model(MODEL_NAME)
    print(f"Registered model '{MODEL_NAME}' already exists.")
except Exception:
    client.create_registered_model(MODEL_NAME)
    print(f"Created registered model '{MODEL_NAME}'.")

model_version = client.create_model_version(
    name=MODEL_NAME,
    source=model_uri,
    run_id=run_id,
)

print("Model version:", model_version.version)

client.transition_model_version_stage(
    name=MODEL_NAME,
    version=model_version.version,
    stage="Staging",
)

print(
    f"Model version {model_version.version} "
    f"moved to Staging."
)