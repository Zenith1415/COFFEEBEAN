"""
COFFEEBEAN — Model Registry Helpers
Phase 6: Register, promote, and manage models in MLflow Model Registry.

Usage:
    from src.deployment.registry import register_model, promote_to_production
"""

import os
import logging
import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

MODEL_NAME = "COFFEEBEAN_ANC"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "https://dagshub.com/Zenith1415/COFFEEBEAN.mlflow")


def get_client() -> MlflowClient:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return MlflowClient()


def register_model(run_id: str, artifact_path: str = "anc_model") -> str:
    """
    Register a trained model from a MLflow run into the Model Registry.

    Args:
        run_id:        MLflow run ID containing the model artifact.
        artifact_path: Path to model artifact within the run.

    Returns:
        Registered model version string.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"runs:/{run_id}/{artifact_path}"

    result = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
    logger.info(f"Registered model '{MODEL_NAME}' version {result.version} from run {run_id}")
    return result.version


def promote_to_staging(version: str) -> None:
    """Move a model version to Staging."""
    client = get_client()
    client.transition_model_version_stage(
        name=MODEL_NAME, version=version, stage="Staging"
    )
    logger.info(f"Model '{MODEL_NAME}' v{version} -> Staging")


def promote_to_production(version: str) -> None:
    """
    Move a model version to Production.
    Archives any existing Production version automatically.
    """
    client = get_client()
    # Archive existing production models
    for mv in client.search_model_versions(f"name='{MODEL_NAME}'"):
        if mv.current_stage == "Production":
            client.transition_model_version_stage(
                name=MODEL_NAME, version=mv.version, stage="Archived"
            )
            logger.info(f"Archived previous Production v{mv.version}")

    client.transition_model_version_stage(
        name=MODEL_NAME, version=version, stage="Production"
    )
    logger.info(f"Model '{MODEL_NAME}' v{version} -> Production")


def get_production_model_uri() -> str:
    """Return the URI for the current Production model."""
    return f"models:/{MODEL_NAME}/Production"


def get_latest_staging_version() -> str | None:
    """Return the latest Staging model version, or None."""
    client = get_client()
    versions = client.get_latest_versions(MODEL_NAME, stages=["Staging"])
    return versions[0].version if versions else None


def list_model_versions() -> list:
    """List all registered model versions with their stages."""
    client = get_client()
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    return [
        {
            "version": mv.version,
            "stage":   mv.current_stage,
            "run_id":  mv.run_id,
            "status":  mv.status,
        }
        for mv in sorted(versions, key=lambda x: int(x.version))
    ]


def tag_run(run_id: str, key: str, value: str) -> None:
    """Add a tag to an MLflow run."""
    get_client().set_tag(run_id, key, value)
