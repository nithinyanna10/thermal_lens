#!/bin/bash
# Quick download script for KAIST dataset

echo "🔥 Thermal Lens - Dataset Download Helper"
echo "=========================================="
echo ""

# Check if huggingface-hub is installed
if python -c "import huggingface_hub" 2>/dev/null; then
    echo "✅ Hugging Face Hub is installed"
    echo ""
    echo "Attempting to download from Hugging Face..."
    source venv/bin/activate
    python download_kaist.py --use_huggingface
else
    echo "📥 Download Options:"
    echo ""
    echo "1. Install Hugging Face Hub and try automatic download:"
    echo "   pip install huggingface-hub"
    echo "   python download_kaist.py --use_huggingface"
    echo ""
    echo "2. Download FLIR ADAS (easier, smaller, no approval):"
    echo "   Visit: https://www.flir.com/oem/adas/adas-dataset-form/"
    echo "   Then: python download_flir_alternative.py"
    echo ""
    echo "3. Manual KAIST download:"
    echo "   Visit: http://multispectral.kaist.ac.kr/pedestrian/"
    echo "   Register, download, extract to data/raw/"
    echo "   Then: python download_kaist.py --skip_download"
    echo ""
fi

