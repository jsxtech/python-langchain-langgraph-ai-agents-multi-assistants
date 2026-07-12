#!/bin/bash
set -e

echo "Setting up Python LangChain LangGraph AI Agents..."

# Create virtual environment
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Virtual environment created."
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create artifacts directory
mkdir -p artifacts

# Check for API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo ""
    echo "WARNING: OPENAI_API_KEY not set."
    echo "Export it before running agents:"
    echo "  export OPENAI_API_KEY='your-key-here'"
    echo ""
    echo "Or create a .env file:"
    echo "  echo 'OPENAI_API_KEY=your-key-here' > .env"
fi

echo "Setup complete! Activate with: source venv/bin/activate"
