#!/bin/bash
# Simple script to run the MediaPipe Live Segmentation application

# Navigate to script directory
cd "$(dirname "$0")"

# Extract textures.zip if textures directory doesn't exist
if [ ! -d "textures" ] && [ -f "textures.zip" ]; then
    echo "📦 Extracting textures..."
    unzip -q textures.zip
    echo "✅ Textures extracted"
fi

# Check if virtual environment exists, create if not
if [ ! -d "venv" ]; then
    echo "🔧 Creating virtual environment..."
    
    # Check for Python 3.11 (preferred)
    if command -v python3.11 &> /dev/null; then
        PYTHON_CMD=python3.11
    elif command -v python3 &> /dev/null; then
        # Check Python version
        PYTHON_VERSION=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
        MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
        MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
        
        if [ "$MAJOR" -gt 3 ] || ([ "$MAJOR" -eq 3 ] && [ "$MINOR" -ge 11 ]); then
            PYTHON_CMD=python3
        else
            echo "❌ Error: Python 3.11+ required. Found: $(python3 --version)"
            echo "   Install Python 3.11: brew install python@3.11"
            exit 1
        fi
    else
        echo "❌ Error: Python 3.11+ not found"
        echo "   Install Python 3.11: brew install python@3.11"
        exit 1
    fi
    
    $PYTHON_CMD -m venv venv
    echo "✅ Virtual environment created"
    
    # Activate and install dependencies
    source venv/bin/activate
    echo "📥 Installing dependencies (this may take a few minutes)..."
    pip install --upgrade pip --quiet
    pip install -r requirements.txt --quiet
    echo "✅ Dependencies installed"
else
    # Activate existing virtual environment
    source venv/bin/activate
fi

# Run the application
python mediapipe_live_segmentation.py

