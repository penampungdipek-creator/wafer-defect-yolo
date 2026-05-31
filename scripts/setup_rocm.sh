#!/bin/bash
# ROCm environment setup for wafer defect detection
set -e

echo "=== Setting up ROCm environment ==="

# Detect GPU architecture
GPU_ARCH=$(rocm-smi --showproductname 2>/dev/null | grep -oP "gfx\d+" || echo "gfx1100")
echo "Detected GPU architecture: $GPU_ARCH"

# Set environment variables
export HSA_OVERRIDE_GFX_VERSION="${GPU_ARCH#gfx}"
export HSA_ENABLE_SDMA=0
export PYTORCH_HIP_ALLOC_CONF="garbage_collection_threshold:0.8,max_split_size_mb:512"
export HIP_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0
export GPU_MAX_HW_QUEUES=8

# GPU-specific overrides
case "$GPU_ARCH" in
    gfx1100)  # RX 7900 XTX
        export HSA_OVERRIDE_GFX_VERSION="11.0.0"
        ;;
    gfx1030)  # RX 6800/6900 XT
        export HSA_OVERRIDE_GFX_VERSION="10.3.0"
        ;;
    gfx90a)   # MI250
        export HSA_OVERRIDE_GFX_VERSION="9.0.0"
        ;;
    gfx90c)   # Integrated GPU
        export HSA_OVERRIDE_GFX_VERSION="9.0.0"
        ;;
esac

# Install PyTorch with ROCm
echo "Installing PyTorch with ROCm support..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2

# Verify installation
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA/ROCm available: {torch.cuda.is_available()}')
if hasattr(torch.version, 'hip'):
    print(f'ROCm/HIP version: {torch.version.hip}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"

echo "=== ROCm setup complete ==="
