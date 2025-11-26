#!/bin/bash
# Thermal Lens v0.1 - Setup Script

echo "🔥 Thermal Lens v0.1 - Setup"
echo "=============================="

# Create necessary directories
echo "Creating directories..."
mkdir -p data/raw/rgb
mkdir -p data/raw/thermal
mkdir -p data/processed/rgb
mkdir -p data/processed/thermal
mkdir -p checkpoints
mkdir -p model/outputs

# Check Python version
echo "Checking Python version..."
python3 --version

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Download KAIST dataset to data/raw/"
echo "2. Run: python utils/align_kaist.py --raw_rgb data/raw/rgb --raw_thermal data/raw/thermal"
echo "3. Run: python model/train.py"
echo ""

