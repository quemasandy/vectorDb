#!/bin/bash

# Vector Database Learning Environment Setup Script
# This script sets up and activates your Python virtual environment

echo "🚀 Setting up Vector Database Learning Environment..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Creating it..."
    python3 -m venv venv
    echo "✅ Virtual environment created!"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing vector database dependencies..."
pip install -r requirements.txt

echo ""
echo "🎉 Environment setup complete!"
echo ""
echo "To activate your environment in the future, run:"
echo "   source venv/bin/activate"
echo ""
echo "To deactivate when you're done:"
echo "   deactivate"
echo ""
echo "🎮 Ready to start Quest 1: Vector Playground!"
echo "   Open roadMap.md and begin your hero's journey!"