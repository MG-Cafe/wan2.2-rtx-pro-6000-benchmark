#!/bin/bash
set -x

echo "=== I2V Benchmarks Only ==="
echo "SGLang will automatically download the Wan2.2-I2V-A14B-Diffusers model."

echo ""
echo "============================================="
echo "=== Downloading test image for I2V ==="
echo "============================================="
mkdir -p assets
# Download a sample image for I2V benchmarks
curl -sL "https://raw.githubusercontent.com/sgl-project/sglang/main/assets/logo.png" -o assets/logo.png || \
  python3 -c "
from PIL import Image
img = Image.new('RGB', (256, 256), color='blue')
img.save('assets/logo.png')
"
ls -la assets/logo.png

echo ""
echo "=============================================="
echo "=== BENCHMARK 1: Image-to-Video on 1 GPU ==="
echo "=============================================="
sglang generate --model-path Wan-AI/Wan2.2-I2V-A14B-Diffusers --image-path assets/logo.png --dit-layerwise-offload false --text-encoder-cpu-offload false --vae-cpu-offload false --pin-cpu-memory --dit-cpu-offload false --prompt "A curious raccoon" --save-output --num-gpus 1 --num-frames 81

echo ""
echo "==============================================="
echo "=== BENCHMARK 2: Image-to-Video on 4 GPUs ==="
echo "==============================================="
sglang generate --model-path Wan-AI/Wan2.2-I2V-A14B-Diffusers --image-path assets/logo.png --dit-layerwise-offload false --text-encoder-cpu-offload false --vae-cpu-offload false --pin-cpu-memory --dit-cpu-offload false --prompt "A curious raccoon" --save-output --num-gpus 4 --tp-size 4 --num-frames 93

echo ""
echo "============================================"
echo "=== I2V BENCHMARKS COMPLETE ==="
echo "============================================"
