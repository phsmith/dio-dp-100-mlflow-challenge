"""Train a linear regression model and log artifacts/metrics to MLflow."""

import argparse
import os
import pathlib
import sys

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split

from app.settings import logger

DEFAULT_DATA_PATH = "data/ice_cream_sales.csv"
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
DEFAULT_EXPERIMENT_NAME = "Ice_Cream_Sales_Prediction"
FEATURE_COLUMN = "Temperature"
TARGET_COLUMN = "Sales"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and track ice cream sales model.")
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH, help="Path to input CSV file.")
    parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help="Proportion of test data in train/test split (0 < value < 1).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Random seed for reproducible train/test split.",
    )
    parser.add_argument(
        "--experiment-name",
        default=DEFAULT_EXPERIMENT_NAME,
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=0,
        help="Number of cross-validation folds. Use 0 to disable CV.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 0 < args.test_size < 1:
        raise ValueError(f"--test-size must be between 0 and 1, got {args.test_size}.")
    if args.cv_folds < 0:
        raise ValueError(f"--cv-folds must be >= 0, got {args.cv_folds}.")


def configure_mlflow(experiment_name: str) -> None:
    azure_uri = os.getenv("AZURE_MLFLOW_URI")
    if azure_uri:
        logger.info("Azure ML tracking URI found. Setting MLflow tracking URI.")
        mlflow.set_tracking_uri(azure_uri)

    mlflow.set_experiment(experiment_name)


def load_and_validate_data(data_path: str) -> tuple[pd.DataFrame, pd.Series]:
    path = pathlib.Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Input data file not found: {data_path}")

    df = pd.read_csv(path)

    required_columns = {FEATURE_COLUMN, TARGET_COLUMN}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required column(s): {missing}")

    data = df[[FEATURE_COLUMN, TARGET_COLUMN]].copy()
    data[FEATURE_COLUMN] = pd.to_numeric(data[FEATURE_COLUMN], errors="coerce")
    data[TARGET_COLUMN] = pd.to_numeric(data[TARGET_COLUMN], errors="coerce")

    invalid_rows = data.isna().any(axis=1).sum()
    if invalid_rows:
        logger.warning(
            "Dropping %s row(s) with missing/non-numeric feature or target values.",
            invalid_rows,
        )
        data = data.dropna()

    if data.empty:
        raise ValueError("No valid rows left after data validation.")

    X = data[[FEATURE_COLUMN]]
    y = data[TARGET_COLUMN]
    return X, y


def evaluate_model(model: LinearRegression, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_test, y_pred)
    return {"mse": float(mse), "rmse": rmse, "r2": float(r2)}


def maybe_run_cross_validation(
    model: LinearRegression,
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int,
) -> dict[str, float]:
    if cv_folds == 0:
        return {}
    if cv_folds < 2:
        raise ValueError("--cv-folds must be >= 2 when enabled.")
    if cv_folds > len(X):
        raise ValueError(
            f"--cv-folds ({cv_folds}) cannot be greater than number of samples ({len(X)})."
        )

    neg_mse_scores = cross_val_score(
        model,
        X,
        y,
        cv=cv_folds,
        scoring="neg_mean_squared_error",
    )
    rmse_scores = np.sqrt(-neg_mse_scores)

    return {
        "cv_rmse_mean": float(rmse_scores.mean()),
        "cv_rmse_std": float(rmse_scores.std()),
    }


def read_pip_requirements(requirements_path: pathlib.Path) -> list[str] | None:
    if not requirements_path.exists():
        return None
    with requirements_path.open("r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip() and not line.startswith("#")]


def log_model_with_fallback(model: LinearRegression, pip_requirements: list[str] | None) -> None:
    try:
        mlflow.sklearn.log_model(
            sk_model=model,
            name="linear_regression_model",
            serialization_format="skops",
            pip_requirements=pip_requirements,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to log model with skops serialization (%s). Falling back to default.",
            exc,
        )
        mlflow.sklearn.log_model(
            sk_model=model,
            name="linear_regression_model",
            pip_requirements=pip_requirements,
        )


def main() -> None:
    args = parse_args()
    validate_args(args)
    configure_mlflow(args.experiment_name)

    logger.info("Starting MLflow run.")
    with mlflow.start_run():
        with mlflow.start_span(name="train_pipeline") as pipeline_span:
            pipeline_span.set_inputs(
                {
                    "data_path": args.data_path,
                    "test_size": args.test_size,
                    "random_state": args.random_state,
                    "cv_folds": args.cv_folds,
                }
            )

            with mlflow.start_span(name="load_and_validate_data") as span:
                X, y = load_and_validate_data(args.data_path)
                span.set_outputs({"rows": len(X), "columns": list(X.columns)})

            with mlflow.start_span(name="split_data") as span:
                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=args.test_size,
                    random_state=args.random_state,
                )
                span.set_outputs(
                    {
                        "train_rows": len(X_train),
                        "test_rows": len(X_test),
                    }
                )

            mlflow.log_params(
                {
                    "data_path": args.data_path,
                    "feature_column": FEATURE_COLUMN,
                    "target_column": TARGET_COLUMN,
                    "test_size": args.test_size,
                    "random_state": args.random_state,
                    "cv_folds": args.cv_folds,
                }
            )

            logger.info("Training Linear Regression model.")
            with mlflow.start_span(name="fit_model") as span:
                model = LinearRegression()
                model.fit(X_train, y_train)
                span.set_outputs({"coef": float(model.coef_[0]), "intercept": float(model.intercept_)})

            logger.info("Evaluating model on holdout set.")
            with mlflow.start_span(name="evaluate_model") as span:
                metrics = evaluate_model(model, X_test, y_test)
                cv_metrics = maybe_run_cross_validation(model, X, y, args.cv_folds)
                all_metrics = {**metrics, **cv_metrics}
                span.set_outputs(all_metrics)

            mlflow.log_metrics(all_metrics)
            logger.info("Evaluation results: %s", all_metrics)

            with mlflow.start_span(name="log_model") as span:
                pip_requirements = read_pip_requirements(pathlib.Path("requirements.txt"))
                log_model_with_fallback(model, pip_requirements)
                span.set_outputs({"model_name": "linear_regression_model"})

            run_id = mlflow.active_run().info.run_id
            pipeline_span.set_outputs({"run_id": run_id, "metrics": all_metrics})
            logger.info("MLflow run completed. Run ID: %s", run_id)
            logger.info(
                "To visualize, run 'mlflow ui' and open experiment '%s'.",
                args.experiment_name,
            )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")
        sys.exit(130)
