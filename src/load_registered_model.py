import mlflow
import mlflow.sklearn

MODEL_URI = "models:/iris-classifier-prod/Staging"

model = mlflow.sklearn.load_model(MODEL_URI)

print("Registered model loaded successfully!")
print("Model type:", type(model))
