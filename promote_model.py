import mlflow
import dagshub
import json
from mlflow import MlflowClient

# Initialize Dagshub and MLflow
dagshub.init(repo_owner='tushar-oclock', repo_name='uber-demand-prediction', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/tushar-oclock/uber-demand-prediction.mlflow")

# Model metadata
registered_model_name = 'uber_demand_prediction_model'
stage = "Staging"
promotion_stage = "Production"

# Initialize MLflow client
client = MlflowClient()

# Get latest version in staging
latest_versions = client.get_latest_versions(name=registered_model_name, stages=[stage])

# Check if staging model exists
if not latest_versions:
    print(f"No model found in '{stage}' stage for model '{registered_model_name}'.")
    # Optional: list all available versions and their stages
    all_versions = client.get_latest_versions(name=registered_model_name)
    if all_versions:
        print("Available model versions:")
        for v in all_versions:
            print(f" - Version {v.version} in stage '{v.current_stage}'")
    else:
        print("No versions found at all for this model.")
    exit(1)

# Promote model to production
latest_model_version_staging = latest_versions[0].version

model_version_prod = client.transition_model_version_stage(
    name=registered_model_name,
    version=latest_model_version_staging,
    stage=promotion_stage,
    archive_existing_versions=True
)

# Output result
print(f"The model is moved to the {model_version_prod.current_stage} stage with version number {model_version_prod.version}")
