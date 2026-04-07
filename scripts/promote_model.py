import os
import mlflow
from mlflow.tracking import MlflowClient


def promote_model():
    
    dagshub_token = os.getenv("CAPSTONE_TEST")
    if not dagshub_token:
        raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    dagshub_url = "https://dagshub.com"
    repo_owner = "babatundejulius911"
    repo_name = "end-to-end-Nlp-Project"
    # Set up MLflow tracking URI
    mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

    client = MlflowClient()
    model_name = "my_model"

    # 2: GET ALL VERSIONS (ROBUST)
    versions = client.search_model_versions(f"name='{model_name}'")

    if not versions:
        raise Exception(f"No versions found for model '{model_name}'")

    # Convert to list (MLflow returns generator sometimes)
    versions = list(versions)

    # 3: FIND STAGING VERSION SAFELY
    staging_versions = [v for v in versions if v.current_stage == "Staging"]

    if not staging_versions:
        raise Exception("No model in Staging stage to promote")

    # Pick latest staging version (highest version number)
    latest_staging = max(staging_versions, key=lambda v: int(v.version))
    staging_version = latest_staging.version

    print(f"Latest Staging Version: {staging_version}")

    # 4: ARCHIVE CURRENT PRODUCTION SAFELY
    production_versions = [v for v in versions if v.current_stage == "Production"]

    for v in production_versions:
        print(f"Archiving Production version: {v.version}")
        client.transition_model_version_stage(
            name=model_name,
            version=v.version,
            stage="Archived"
        )

    # 5: PROMOTE TO PRODUCTION
    client.transition_model_version_stage(
        name=model_name,
        version=staging_version,
        stage="Production"
    )

    print(f Model version {staging_version} promoted to Production")


if __name__ == "__main__":
    promote_model()