"""Run exploratory analysis and visualization for ice cream sales data."""

import argparse
import pathlib

import matplotlib.pyplot as plt
import pandas as pd

from app.settings import logger

DEFAULT_DATA_PATH = "data/ice_cream_sales.csv"
DEFAULT_PLOT_PATH = "data/temperature_vs_sales.png"
FEATURE_COLUMN = "Temperature"
TARGET_COLUMN = "Sales"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run exploratory analysis for ice cream sales data.")
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH, help="Path to input CSV data.")
    parser.add_argument("--plot-path", default=DEFAULT_PLOT_PATH, help="Path to output plot image.")
    return parser.parse_args()


def load_and_validate_data(data_path: str) -> pd.DataFrame:
    path = pathlib.Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Input data file not found: {data_path}")

    df = pd.read_csv(path)
    required_columns = {FEATURE_COLUMN, TARGET_COLUMN}
    missing = required_columns.difference(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Missing required column(s): {missing_str}")

    data = df[[FEATURE_COLUMN, TARGET_COLUMN]].copy()
    data[FEATURE_COLUMN] = pd.to_numeric(data[FEATURE_COLUMN], errors="coerce")
    data[TARGET_COLUMN] = pd.to_numeric(data[TARGET_COLUMN], errors="coerce")

    invalid_rows = data.isna().any(axis=1).sum()
    if invalid_rows:
        logger.warning(
            "Dropping %s row(s) with missing/non-numeric '%s' or '%s'.",
            invalid_rows,
            FEATURE_COLUMN,
            TARGET_COLUMN,
        )
        data = data.dropna()

    if data.empty:
        raise ValueError("No valid rows left after cleaning.")

    return data


def generate_plot(data: pd.DataFrame, plot_path: str) -> None:
    output_path = pathlib.Path(plot_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(data[FEATURE_COLUMN], data[TARGET_COLUMN], alpha=0.7, edgecolors="k")
    ax.set_title("Relationship between Temperature and Ice Cream Sales", fontsize=16)
    ax.set_xlabel("Temperature (C)", fontsize=12)
    ax.set_ylabel("Sales (Units)", fontsize=12)
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Saved plot to '%s'.", output_path)


def main() -> None:
    args = parse_args()
    logger.info("Loading data from '%s'.", args.data_path)
    data = load_and_validate_data(args.data_path)

    logger.info("--- Descriptive Statistics ---\n%s", data.describe().to_string())
    logger.info("--- Correlation Matrix ---\n%s", data.corr(numeric_only=True).to_string())

    logger.info("Generating plot.")
    generate_plot(data, args.plot_path)
    logger.info("Exploratory analysis complete.")


if __name__ == "__main__":
    main()
