#!/bin/bash
# Setup script for VoxCPM installation and running

set -e  # Exit on error

echo "🔧 Setting up VoxCPM..."

# Activate virtual environment if it exists
if [ -f "bin/activate" ]; then
    echo "📦 Activating virtual environment..."
    source bin/activate
else
    echo "⚠️  No virtual environment found. Creating one..."
    python3 -m venv .
    source bin/activate
fi

# Upgrade pip and build tools
echo "⬆️  Upgrading pip, setuptools, and wheel..."
python3 -m pip install --upgrade pip setuptools wheel

# Set environment variable for setuptools-scm
export SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0

# Install the package in editable mode
echo "📥 Installing voxcpm in editable mode..."
python3 -m pip install -e .

echo "✅ Installation complete!"
echo ""
echo "🚀 You can now run the app with:"
echo "   python3 app.py"
echo ""
echo "Or activate the virtual environment first:"
echo "   source bin/activate"
echo "   python app.py"

