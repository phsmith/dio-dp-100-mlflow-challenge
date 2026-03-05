"""FastAPI entrypoint for ice cream sales predictions."""

import math
import os
import threading
from typing import Any

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, status

from app.settings import logger

# --- Application Setup ---
app = FastAPI(title="Ice Cream Sales Prediction API")

# --- Configuration ---
DEFAULT_EXPERIMENT_NAME = "Ice_Cream_Sales_Prediction"
DEFAULT_MODEL_ARTIFACT_PATH = "linear_regression_model"
FEATURE_COLUMN = "Temperature"
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT_NAME)
MODEL_ARTIFACT_PATH = os.getenv("MLFLOW_MODEL_ARTIFACT_PATH", DEFAULT_MODEL_ARTIFACT_PATH)
MIN_R2_ENV = os.getenv("MLFLOW_MIN_R2")

try:
    MIN_R2 = float(MIN_R2_ENV) if MIN_R2_ENV is not None else None
except ValueError:
    logger.warning("Invalid MLFLOW_MIN_R2 value '%s'. Ignoring threshold.", MIN_R2_ENV)
    MIN_R2 = None

# --- Model state ---
MODEL_LOADED: Any | None = None
MODEL_RUN_ID: str | None = None
MODEL_LOCK = threading.Lock()


def configure_mlflow_tracking() -> None:
    azure_uri = os.getenv("AZURE_MLFLOW_URI")
    if azure_uri:
        logger.info("Azure ML tracking URI found. Setting MLflow tracking URI.")
        mlflow.set_tracking_uri(azure_uri)


def _build_run_filter(min_r2: float | None) -> str:
    base = "attributes.status = 'FINISHED'"
    if min_r2 is None:
        return base
    return f"{base} and metrics.r2 >= {min_r2}"


def load_model() -> bool:
    """Loads the best model from MLflow for serving."""
    global MODEL_LOADED, MODEL_RUN_ID
    with MODEL_LOCK:
        with mlflow.start_span(name="api_load_model") as span:
            span.set_inputs(
                {
                    "experiment_name": EXPERIMENT_NAME,
                    "model_artifact_path": MODEL_ARTIFACT_PATH,
                    "min_r2": MIN_R2,
                }
            )
            try:
                configure_mlflow_tracking()
                client = mlflow.tracking.MlflowClient()
                experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
                if not experiment:
                    raise RuntimeError(f"Experiment '{EXPERIMENT_NAME}' not found.")

                runs = client.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    filter_string=_build_run_filter(MIN_R2),
                    order_by=["metrics.r2 DESC", "start_time DESC"],
                    max_results=1,
                )
                if not runs:
                    raise RuntimeError(
                        f"No FINISHED runs found for experiment '{EXPERIMENT_NAME}' "
                        f"with threshold MLFLOW_MIN_R2={MIN_R2}."
                    )

                MODEL_RUN_ID = runs[0].info.run_id
                model_uri = f"runs:/{MODEL_RUN_ID}/{MODEL_ARTIFACT_PATH}"
                logger.info("Loading model from URI: %s", model_uri)
                MODEL_LOADED = mlflow.pyfunc.load_model(model_uri)
                span.set_outputs({"loaded": True, "model_run_id": MODEL_RUN_ID})
                logger.info("Model loaded successfully from run %s.", MODEL_RUN_ID)
                return True
            except Exception:  # noqa: BLE001
                logger.exception("Failed to load model for serving.")
                MODEL_LOADED = None
                MODEL_RUN_ID = None
                span.set_outputs({"loaded": False})
                return False


def _ensure_model_loaded() -> bool:
    if MODEL_LOADED is not None:
        return True
    return load_model()

# --- API Endpoints ---
@app.get("/")
def home() -> dict[str, Any]:
    """Main endpoint to check API status."""
    return {
        "message": "Ice Cream Sales Prediction API is running.",
        "model_loaded": MODEL_LOADED is not None,
        "experiment_name": EXPERIMENT_NAME,
        "model_run_id": MODEL_RUN_ID,
    }


@app.get("/predict")
def predict(temperature: str | None = Query(default=None)) -> dict[str, float]:
    """Endpoint to make sales predictions."""
    temp_str = temperature
    with mlflow.start_span(name="api_predict") as span:
        span.set_inputs({"temperature_raw": temp_str})

        if not _ensure_model_loaded():
            span.set_outputs({"status_code": 500, "error": "model_not_loaded"})
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Model is not loaded. Check server logs.",
            )

        if temp_str is None:
            span.set_outputs({"status_code": 400, "error": "missing_temperature"})
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Parameter 'temperature' not found in URL.",
            )

        try:
            temperature_value = float(temp_str)
            if not math.isfinite(temperature_value):
                raise ValueError("temperature must be a finite number")

            data = pd.DataFrame([[temperature_value]], columns=[FEATURE_COLUMN])
            prediction = MODEL_LOADED.predict(data)
            prediction_value = round(float(prediction[0]), 2)
            span.set_outputs(
                {
                    "status_code": 200,
                    "temperature": temperature_value,
                    "sales_prediction": prediction_value,
                    "model_run_id": MODEL_RUN_ID,
                }
            )
            return {
                "provided_temperature": temperature_value,
                "sales_prediction": prediction_value,
            }
        except ValueError:
            span.set_outputs({"status_code": 400, "error": "invalid_temperature"})
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"'{temp_str}' is not a valid value for 'temperature'.",
            )
        except Exception:  # noqa: BLE001
            logger.exception("Unexpected error during prediction.")
            span.set_outputs({"status_code": 500, "error": "internal_error"})
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An internal error occurred during prediction.",
            )


@app.post("/reload-model")
def reload_model() -> dict[str, str]:
    """Reloads the model without restarting the server."""
    with mlflow.start_span(name="api_reload_model") as span:
        loaded = load_model()
        if not loaded:
            span.set_outputs({"status_code": 500, "loaded": False})
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to reload model. Check server logs.",
            )
        span.set_outputs({"status_code": 200, "loaded": True, "model_run_id": MODEL_RUN_ID})
        return {"message": "Model reloaded successfully.", "model_run_id": MODEL_RUN_ID or ""}
