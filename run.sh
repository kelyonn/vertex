#!/bin/bash
# Project Vertex v4 — JARVIS Hologram Workbench

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

echo ">> STARTING PROJECT VERTEX v4.0"
echo ">> ----------------------------"

# Detect Python 3.11 first, fall back to python3
PYTHON_CMD="python3.11"
if ! command -v $PYTHON_CMD &> /dev/null; then
    PYTHON_CMD="python3"
    if ! command -v $PYTHON_CMD &> /dev/null; then
        echo ">> ERROR: Python 3.11+ not found. Please install Python 3.11."
        exit 1
    fi
fi
echo ">> Python: $PYTHON_CMD ($(${PYTHON_CMD} --version 2>&1))"

# Create virtual environment if needed
if [ ! -d "venv" ]; then
    echo ">> Creating virtual environment..."
    $PYTHON_CMD -m venv venv
fi

VENV_PYTHON="$DIR/venv/bin/python"

# Install / sync dependencies (pip is smart about skipping up-to-date packages)
echo ">> Syncing dependencies..."
"$VENV_PYTHON" -m pip install -q --upgrade pip
"$VENV_PYTHON" -m pip install -q -r requirements.txt

# Download MediaPipe hand landmarker model if missing
if [ ! -f "src/hand_landmarker.task" ]; then
    echo ">> Downloading MediaPipe hand landmarker model..."
    curl -L -o src/hand_landmarker.task \
      https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
fi

# Create models directory if missing
mkdir -p models

echo ">> Launching application..."
"$VENV_PYTHON" src/main.py "$@"

echo ">> Shutdown complete."
