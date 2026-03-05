"""Generate synthetic ice cream sales data."""

import argparse
import pathlib

import numpy as np
import pandas as pd

from app.settings import logger

DEFAULT_OUTPUT_PATH = "data/ice_cream_sales.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic ice cream sales dataset.")
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH, help="Output CSV path.")
    parser.add_argument("--num-days", type=int, default=100, help="Number of daily rows to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--temp-min", type=float, default=20.0, help="Minimum baseline temperature.")
    parser.add_argument("--temp-max", type=float, default=38.0, help="Maximum baseline temperature.")
    parser.add_argument("--base-sales", type=float, default=25.0, help="Baseline sales value.")
    parser.add_argument("--temp-factor", type=float, default=5.5, help="Sales impact per temperature unit.")
    parser.add_argument("--sales-noise", type=float, default=30.0, help="Std-dev noise added to sales.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_days <= 0:
        raise ValueError(f"--num-days must be > 0, got {args.num_days}.")
    if args.temp_max <= args.temp_min:
        raise ValueError("--temp-max must be greater than --temp-min.")
    if args.sales_noise < 0:
        raise ValueError("--sales-noise must be >= 0.")


def generate_dataset(args: argparse.Namespace) -> pd.DataFrame:
    rng = np.random.default_rng(args.seed)
    days = np.arange(args.num_days)
    # Build a smooth seasonal curve: sin() oscillates in [-1, 1], then we shift/scale it to [0, 1].
    # This makes temperatures move between temp_min and temp_max over an approximately 90-day cycle.
    temperatures = args.temp_min + (args.temp_max - args.temp_min) * 0.5 * (
        1 + np.sin(2 * np.pi * days / 90)
    )
    # Add day-to-day weather randomness around the seasonal baseline.
    temperatures += rng.normal(0, 1.5, args.num_days)
    # Keep temperatures in a plausible interval so outliers do not dominate the synthetic data.
    temperatures = np.clip(temperatures, args.temp_min - 2, args.temp_max + 2)

    # Sales are modeled as a linear response to temperature:
    # warmer days tend to increase demand by `temp_factor` per degree above temp_min.
    sales = args.base_sales + (temperatures - args.temp_min) * args.temp_factor
    # Add stochastic demand variation (events, weekdays, etc.) not explained by temperature alone.
    sales += rng.normal(0, args.sales_noise, args.num_days)
    # Round to whole units and enforce a non-trivial minimum sales floor.
    sales = np.maximum(np.round(sales), 10).astype(int)

    # Final dataset schema used by training and inference code.
    return pd.DataFrame({"Temperature": temperatures, "Sales": sales})


def save_dataset(df: pd.DataFrame, output_path: str) -> None:
    path = pathlib.Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Data generated and saved to '%s'.", path)


def main() -> None:
    args = parse_args()
    validate_args(args)
    df = generate_dataset(args)
    save_dataset(df, args.output_path)
    logger.info("Sample rows:\n%s", df.head().to_string(index=False))


if __name__ == "__main__":
    main()
