#!/bin/bash
set -x

echo "============================================="
echo "=== Downloading test image for I2V ==="
echo "============================================="
mkdir -p assets
curl -sL "https://raw.githubusercontent.com/sgl-project/sglang/main/assets/logo.png" -o assets/logo.png || \
  python3 -c "
from PIL import Image
img = Image.new('RGB', (256, 256), color='blue')
img.save('assets/logo.png')
"

echo "==============================================="
echo "=== BENCHMARK: Image-to-Video on 8 GPUs ==="
echo "==============================================="
sglang generate --model-path Wan-AI/Wan2.2-I2V-A14B-Diffusers --image-path assets/logo.png --dit-layerwise-offload false --text-encoder-cpu-offload false --vae-cpu-offload false --pin-cpu-memory --dit-cpu-offload false --prompt "A curious raccoon" --save-output --num-gpus 8 --tp-size 8 --num-frames 93

echo ""
echo "=== I2V 8-GPU BENCHMARK COMPLETE ==="
