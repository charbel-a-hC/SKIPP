#!/bin/bash
echo "Starting application"

# Navigate to the project directory
cd /SKIPP

# Run the Python script using Poetry
poetry run python3 train_bc.py --config ./configs/train_bc_sweep.config.yaml

