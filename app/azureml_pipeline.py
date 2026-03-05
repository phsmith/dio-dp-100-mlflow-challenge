import argparse
import os
import pathlib
import sys
from typing import Any

from app.settings import logger

DEFAULT_EXPERIMENT_NAME = "Ice_Cream_Sales_Prediction"
DEFAULT_COMPUTE = "cpu-cluster"
DEFAULT_JOB_NAME = "ice-cream-sales-pipeline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create and optionally submit an Azure ML pipeline job."
    )
    parser.add_argument("--subscription-id", default=os.getenv("AZURE_SUBSCRIPTION_ID"))
    parser.add_argument("--resource-group", default=os.getenv("AZURE_RESOURCE_GROUP"))
    parser.add_argument("--workspace-name", default=os.getenv("AZURE_ML_WORKSPACE"))
    parser.add_argument("--compute", default=os.getenv("AZURE_ML_COMPUTE", DEFAULT_COMPUTE))
    parser.add_argument(
        "--experiment-name",
        default=os.getenv("AZURE_ML_EXPERIMENT", DEFAULT_EXPERIMENT_NAME),
    )
    parser.add_argument(
        "--job-name",
        default=os.getenv("AZURE_ML_PIPELINE_JOB_NAME", DEFAULT_JOB_NAME),
    )
    parser.add_argument(
        "--source-dir",
        default=".",
        help="Project root that contains app/, data/, and requirements.txt.",
    )
    parser.add_argument(
        "--skip-submit",
        action="store_true",
        help="Build the pipeline object but do not submit it to Azure ML.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    missing = [
        name
        for name, value in [
            ("subscription-id", args.subscription_id),
            ("resource-group", args.resource_group),
            ("workspace-name", args.workspace_name),
            ("compute", args.compute),
        ]
        if not value
    ]
    if missing:
        formatted = ", ".join(missing)
        raise ValueError(
            f"Missing required Azure ML settings: {formatted}. "
            "Set CLI args or env vars (AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, "
            "AZURE_ML_WORKSPACE, AZURE_ML_COMPUTE)."
        )

    src = pathlib.Path(args.source_dir)
    if not src.exists():
        raise FileNotFoundError(f"--source-dir does not exist: {args.source_dir}")


def _import_azure_sdk() -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    try:
        from azure.ai.ml import Input, MLClient, Output, command, dsl
        from azure.ai.ml.constants import AssetTypes
        from azure.identity import DefaultAzureCredential

        return MLClient, DefaultAzureCredential, Input, Output, AssetTypes, command, dsl
    except ImportError as exc:
        raise ImportError(
            "Azure ML dependencies are missing. Install with:\n"
            "  pip install azure-ai-ml azure-identity"
        ) from exc


def build_pipeline(args: argparse.Namespace) -> tuple[Any, Any]:
    MLClient, DefaultAzureCredential, Input, Output, AssetTypes, command, dsl = _import_azure_sdk()

    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=args.subscription_id,
        resource_group_name=args.resource_group,
        workspace_name=args.workspace_name,
    )

    source_dir = str(pathlib.Path(args.source_dir).resolve())
    generate_data_step = command(
        name="generate_data",
        display_name="Generate Synthetic Data",
        code=source_dir,
        command="python -m app.generate_data --output-path ${{outputs.generated_data}}",
        outputs={"generated_data": Output(type=AssetTypes.URI_FILE)},
        environment="AzureML-sklearn-1.5:1",
        compute=args.compute,
    )

    explore_data_step = command(
        name="explore_data",
        display_name="Explore Data",
        code=source_dir,
        command=(
            "python -m app.explore_data --data-path ${{inputs.data_path}} "
            "--plot-path data/temperature_vs_sales.png"
        ),
        inputs={"data_path": Input(type=AssetTypes.URI_FILE)},
        environment="AzureML-sklearn-1.5:1",
        compute=args.compute,
    )

    train_step = command(
        name="train_model",
        display_name="Train Model",
        code=source_dir,
        command="python -m app.train --data-path ${{inputs.data_path}} --cv-folds 5",
        inputs={"data_path": Input(type=AssetTypes.URI_FILE)},
        environment="AzureML-sklearn-1.5:1",
        compute=args.compute,
    )

    @dsl.pipeline(
        description="Ice cream sales training pipeline",
        default_compute=args.compute,
    )
    def ice_cream_pipeline() -> None:
        generate_node = generate_data_step()
        explore_node = explore_data_step(data_path=generate_node.outputs.generated_data)
        train_node = train_step(data_path=generate_node.outputs.generated_data)
        explore_node.after(generate_node)
        train_node.after(explore_node)

    return ml_client, ice_cream_pipeline()


def main() -> None:
    args = parse_args()
    validate_args(args)

    ml_client, pipeline_job = build_pipeline(args)
    pipeline_job.display_name = args.job_name
    pipeline_job.experiment_name = args.experiment_name

    if args.skip_submit:
        logger.info(
            "Azure ML pipeline created (not submitted): job_name=%s experiment=%s compute=%s",
            args.job_name,
            args.experiment_name,
            args.compute,
        )
        return

    created_job = ml_client.jobs.create_or_update(pipeline_job)
    logger.info(
        "Azure ML pipeline submitted: name=%s experiment=%s studio_url=%s",
        created_job.name,
        args.experiment_name,
        getattr(created_job, "studio_url", "N/A"),
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to create/submit Azure ML pipeline: %s", exc)
        sys.exit(1)
