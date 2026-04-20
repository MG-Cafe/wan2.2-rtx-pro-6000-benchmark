#!/bin/bash
set -ex

echo "=== T2V Benchmarks Only ==="
echo "SGLang will automatically download the Wan2.2-T2V-A14B-Diffusers model."

echo ""
echo "============================================"
echo "=== BENCHMARK 1: Text-to-Video on 1 GPU ==="
echo "============================================"
sglang generate --model-path Wan-AI/Wan2.2-T2V-A14B-Diffusers  --dit-layerwise-offload false --text-encoder-cpu-offload false --vae-cpu-offload false --pin-cpu-memory --dit-cpu-offload false     --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside." --save-output --num-gpus 1 --num-frames 81

echo ""
echo "============================================="
echo "=== BENCHMARK 2: Text-to-Video on 4 GPUs ==="
echo "============================================="
sglang generate --model-path Wan-AI/Wan2.2-T2V-A14B-Diffusers  --dit-layerwise-offload false --text-encoder-cpu-offload false --vae-cpu-offload false --pin-cpu-memory --dit-cpu-offload false     --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside." --save-output --num-gpus 4  --tp-size 4 --num-frames 93

echo ""
echo "============================================"
echo "=== T2V BENCHMARKS COMPLETE ==="
echo "============================================"
