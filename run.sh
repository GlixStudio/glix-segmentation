#!/bin/bash
# Simple script to run the MediaPipe Live Segmentation application

# Navigate to script directory
cd "$(dirname "$0")"

# Activate virtual environment
source venv/bin/activate

# Run the application
python mediapipe_live_segmentation.py

