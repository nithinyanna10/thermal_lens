#!/bin/bash
# Download FLIR ADAS Dataset v2
# Downloads all 12 parts and combines them

set -e

DOWNLOAD_DIR="$HOME/Downloads/FLIR_ADAS"
EXTRACT_DIR="data/raw/flir"

echo "🔥 FLIR ADAS Dataset Downloader"
echo "================================"
echo ""

# Create download directory
mkdir -p "$DOWNLOAD_DIR"
cd "$DOWNLOAD_DIR"

echo "📥 Downloading FLIR ADAS v2 (12 parts)..."
echo "This may take 30-60 minutes depending on your connection..."
echo ""

# Download all 12 parts
for i in {00..11}; do
    echo "Downloading part $i/11..."
    curl --progress-bar --output "FLIR_ADAS_v2.zip.${i}" \
        "https://adas-dataset-v2.flirconservator.com/dataset/parts/FLIR_ADAS_v2.zip.${i}" || {
        echo "❌ Error downloading part $i"
        echo "You can resume by running this script again"
        exit 1
    }
done

echo ""
echo "✅ All parts downloaded!"
echo ""

# Verify files
echo "🔍 Verifying downloaded files..."
for i in {00..11}; do
    if [ ! -f "FLIR_ADAS_v2.zip.${i}" ]; then
        echo "❌ Missing part: FLIR_ADAS_v2.zip.${i}"
        exit 1
    fi
done

echo "✅ All parts present"
echo ""

# Combine parts
echo "🔗 Combining parts into single zip file..."
cat FLIR_ADAS_v2.zip.* > FLIR_ADAS_v2.zip

# Verify combined file
if [ ! -f "FLIR_ADAS_v2.zip" ]; then
    echo "❌ Failed to combine parts"
    exit 1
fi

echo "✅ Combined successfully"
echo ""

# Extract
echo "📦 Extracting dataset..."
unzip -q FLIR_ADAS_v2.zip

# Move to project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$PROJECT_DIR/$EXTRACT_DIR"

if [ -d "FLIR_ADAS_v2" ]; then
    echo "📁 Moving to project directory..."
    mv FLIR_ADAS_v2/* "$PROJECT_DIR/$EXTRACT_DIR/" 2>/dev/null || {
        # If move fails, try copying
        cp -r FLIR_ADAS_v2/* "$PROJECT_DIR/$EXTRACT_DIR/"
    }
    echo "✅ Dataset ready at: $PROJECT_DIR/$EXTRACT_DIR"
else
    echo "❌ Extraction failed - FLIR_ADAS_v2 directory not found"
    exit 1
fi

echo ""
echo "🎉 Download complete!"
echo ""
echo "Next steps:"
echo "1. Process the dataset:"
echo "   python utils/process_flir.py --input $EXTRACT_DIR/FLIR_ADAS_v2"
echo ""
echo "2. Train the model:"
echo "   python model/train.py --rgb_dir data/processed/rgb --thermal_dir data/processed/thermal"

