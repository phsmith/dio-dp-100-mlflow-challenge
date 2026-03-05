#!/bin/bash

# This script runs the complete Machine Learning pipeline.
# Usage:
#   ./run_pipeline.sh                 # Local pipeline (default)
#   ./run_pipeline.sh --azureml-pipeline  # Create/submit Azure ML pipeline job

# Ensures that the script will stop if any command fails
set -e

RUN_MODE="local"
if [ "${1:-}" = "--azureml-pipeline" ]; then
  RUN_MODE="azure"
fi

echo "======================================="
echo "STARTING THE ML PIPELINE"
echo "======================================="

# Activate the virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

if [ "$RUN_MODE" = "azure" ]; then
  echo -e "\n--- Azure ML Pipeline Mode ---"
  python3 -m app.azureml_pipeline
  echo -e "\n======================================="
  echo "AZURE ML PIPELINE SUBMITTED SUCCESSFULLY"
  echo "======================================="
  exit 0
fi

# Step 1: Generate data
echo -e "\n--- Step 1: Generating Data ---"
python3 -m app.generate_data

# Step 2: Exploratory Analysis
echo -e "\n--- Step 2: Running Exploratory Analysis ---"
python3 -m app.explore_data

# Step 3: Model Training
echo -e "\n--- Step 3: Training the Model ---"
python3 -m app.train

echo -e "\n======================================="
echo "ML PIPELINE COMPLETED SUCCESSFULLY"
echo "======================================="
