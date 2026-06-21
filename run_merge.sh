#!/bin/bash

# Model Merge Quick Start Script
# This script merges DeepHat-V1-7B and Gemma-4-12B-OBLITERATED using linear interpolation

set -e  # Exit on error

echo "=== Model Merge Script ==="
echo "Merging: DeepHat/DeepHat-V1-7B + OBLITERATUS/Gemma-4-12B-OBLITERATED"
echo "Strategy: Linear Interpolation with 0.5 ratio"
echo "Target: ichnixkann/schweinshaxxe"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --quiet torch transformers accelerate huggingface_hub safetensors

# Check for HF token
if [ -z "$HF_TOKEN" ]; then
    echo "Warning: HF_TOKEN environment variable not set."
    echo "Attempting to proceed without authentication..."
    echo "If download fails, set your token: export HF_TOKEN='your_token_here'"
    echo ""
    read -p "Press Enter to continue or Ctrl+C to cancel..."
else
    echo "HF_TOKEN found, proceeding with authentication..."
fi

# Run the merge
echo "Starting merge process..."
echo "This may take several hours depending on your network and hardware..."
echo ""

python simple_merge.py \
    --model1 "DeepHat/DeepHat-V1-7B" \
    --model2 "OBLITERATUS/Gemma-4-12B-OBLITERATED" \
    --output "./merged_model" \
    --ratio 0.5 \
    --token "$HF_TOKEN" \
    --upload "ichnixkann/schweinshaxxe"

echo ""
echo "=== Merge Complete ==="
echo "Check the merged_model directory for results"
echo "If upload was requested, check your Hugging Face repository: https://huggingface.co/ichnixkann/schweinshaxxe"