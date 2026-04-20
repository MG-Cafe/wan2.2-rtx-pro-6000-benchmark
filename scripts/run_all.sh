#!/bin/bash
#
# Wan2.2 SGLang Benchmark - Complete End-to-End Runner
#
# Usage:
#   export PROJECT_ID="your-project-id"
#   export ZONE="europe-west4-b"  # or any zone with G4 capacity
#   bash scripts/run_all.sh
#
# Prerequisites:
#   - Google Cloud SDK installed and authenticated (gcloud auth login)
#   - Sufficient GPU quota for g4-standard-384 in your chosen zone
#   - IAP API enabled (for SSH tunneling)
#
set -euo pipefail

# ================================================================
# Configuration
# ================================================================
PROJECT_ID="${PROJECT_ID:?ERROR: Set PROJECT_ID environment variable}"
ZONE="${ZONE:-europe-west4-b}"
MACHINE_TYPE="g4-standard-384"
IMAGE_PROJECT="ubuntu-os-accelerator-images"
IMAGE_FAMILY="ubuntu-accelerator-2404-amd64-with-nvidia-570"
BOOT_DISK_SIZE="500GB"
DOCKER_IMAGE="lmsysorg/sglang:latest"

VM_T2V="benchmark-g4-t2v-$$"
VM_I2V="benchmark-g4-i2v-$$"

SSH_CMD="gcloud compute ssh --project=${PROJECT_ID} --zone=${ZONE} --tunnel-through-iap"
SCP_CMD="gcloud compute scp --project=${PROJECT_ID} --zone=${ZONE} --tunnel-through-iap"

RESULTS_DIR="$(pwd)/results/run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RESULTS_DIR}"

echo "============================================="
echo "  Wan2.2 SGLang Benchmark - End-to-End"
echo "============================================="
echo "Project:  ${PROJECT_ID}"
echo "Zone:     ${ZONE}"
echo "Machine:  ${MACHINE_TYPE}"
echo "Results:  ${RESULTS_DIR}"
echo "VMs:      ${VM_T2V} (T2V), ${VM_I2V} (I2V)"
echo "============================================="
echo ""

# ================================================================
# Step 1: Create VMs
# ================================================================
echo "[Step 1/7] Creating G4 VMs..."

for VM in ${VM_T2V} ${VM_I2V}; do
  echo "  Creating ${VM}..."
  gcloud compute instances create ${VM} \
    --machine-type=${MACHINE_TYPE} \
    --project=${PROJECT_ID} \
    --zone=${ZONE} \
    --image-project=${IMAGE_PROJECT} \
    --image-family=${IMAGE_FAMILY} \
    --maintenance-policy=TERMINATE \
    --boot-disk-size=${BOOT_DISK_SIZE} \
    --quiet 2>&1 | tee -a "${RESULTS_DIR}/vm_creation.log"
done

echo "  Waiting 60s for VMs to boot..."
sleep 60

# ================================================================
# Step 2: Resize filesystems
# ================================================================
echo "[Step 2/7] Resizing filesystems to use full ${BOOT_DISK_SIZE}..."

for VM in ${VM_T2V} ${VM_I2V}; do
  ${SSH_CMD} ${VM} --command="
    sudo growpart /dev/nvme0n1 1 2>/dev/null || true
    sudo resize2fs /dev/nvme0n1p1 2>/dev/null || true
    echo 'Disk space:' && df -h / | tail -1
  " 2>&1 | tee -a "${RESULTS_DIR}/disk_resize.log"
done

# ================================================================
# Step 3: Install Docker + NVIDIA Container Toolkit
# ================================================================
echo "[Step 3/7] Installing Docker and NVIDIA Container Toolkit on both VMs..."

for VM in ${VM_T2V} ${VM_I2V}; do
  echo "  Setting up ${VM}..."
  ${SSH_CMD} ${VM} --command="
    set -e
    # Verify GPUs
    nvidia-smi

    # Install Docker
    sudo apt-get update -qq
    sudo apt-get install -y -qq ca-certificates curl > /dev/null
    sudo install -m 0755 -d /etc/apt/keyrings
    sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
    sudo chmod a+r /etc/apt/keyrings/docker.asc
    echo \"deb [arch=\$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \$(. /etc/os-release && echo \\\"\${UBUNTU_CODENAME:-\$VERSION_CODENAME}\\\") stable\" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update -qq
    sudo apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin > /dev/null
    sudo systemctl start docker && sudo systemctl enable docker

    # Install NVIDIA Container Toolkit
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg 2>/dev/null
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null
    sudo apt-get update -qq
    sudo apt-get install -y -qq nvidia-container-toolkit > /dev/null
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker

    # Prepare scratch
    sudo mkdir -p /scratch/cache
    echo 'Setup complete for ${VM}'
  " 2>&1 | tee -a "${RESULTS_DIR}/setup_${VM}.log"
done

# ================================================================
# Step 4: Pull Docker image on both VMs
# ================================================================
echo "[Step 4/7] Pulling SGLang Docker image on both VMs (this takes ~5 min)..."

for VM in ${VM_T2V} ${VM_I2V}; do
  ${SSH_CMD} ${VM} --command="sudo docker pull ${DOCKER_IMAGE}" 2>&1 | tee -a "${RESULTS_DIR}/docker_pull_${VM}.log" &
done
wait
echo "  Docker image pulled on both VMs."

# ================================================================
# Step 5: Run T2V benchmarks on VM1
# ================================================================
echo "[Step 5/7] Running T2V benchmarks on ${VM_T2V}..."

${SSH_CMD} ${VM_T2V} --command="
cat > /scratch/t2v_benchmarks.sh << 'SCRIPT'
#!/bin/bash
set -ex

echo '=== T2V 1-GPU (81 frames) ==='
sglang generate --model-path Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --dit-layerwise-offload false --text-encoder-cpu-offload false \
  --vae-cpu-offload false --pin-cpu-memory --dit-cpu-offload false \
  --prompt 'Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline intricate details and the refreshing atmosphere of the seaside.' \
  --save-output --num-gpus 1 --num-frames 81

echo '=== T2V 4-GPU (93 frames, TP=4) ==='
sglang generate --model-path Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --dit-layerwise-offload false --text-encoder-cpu-offload false \
  --vae-cpu-offload false --pin-cpu-memory --dit-cpu-offload false \
  --prompt 'Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline intricate details and the refreshing atmosphere of the seaside.' \
  --save-output --num-gpus 4 --tp-size 4 --num-frames 93

echo '=== T2V 8-GPU (93 frames, TP=8) ==='
sglang generate --model-path Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --dit-layerwise-offload false --text-encoder-cpu-offload false \
  --vae-cpu-offload false --pin-cpu-memory --dit-cpu-offload false \
  --prompt 'Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline intricate details and the refreshing atmosphere of the seaside.' \
  --save-output --num-gpus 8 --tp-size 8 --num-frames 93

echo '=== ALL T2V BENCHMARKS COMPLETE ==='
SCRIPT
chmod +x /scratch/t2v_benchmarks.sh

sudo docker run -d --name t2v-bench --gpus all \
  -v /scratch:/scratch -v /scratch/cache:/root/.cache --ipc=host \
  ${DOCKER_IMAGE} \
  /bin/bash -c '/scratch/t2v_benchmarks.sh 2>&1 | tee /scratch/t2v_output.log'

echo 'T2V benchmarks started in container.'
" 2>&1 | tee -a "${RESULTS_DIR}/t2v_start.log"

# ================================================================
# Step 6: Run I2V benchmarks on VM2
# ================================================================
echo "[Step 6/7] Running I2V benchmarks on ${VM_I2V}..."

${SSH_CMD} ${VM_I2V} --command="
cat > /scratch/i2v_benchmarks.sh << 'SCRIPT'
#!/bin/bash
set -ex

# Download test image
mkdir -p assets
curl -sL 'https://raw.githubusercontent.com/sgl-project/sglang/main/assets/logo.png' -o assets/logo.png || \
  python3 -c \"from PIL import Image; Image.new('RGB',(256,256),'blue').save('assets/logo.png')\"

echo '=== I2V 1-GPU (81 frames) ==='
sglang generate --model-path Wan-AI/Wan2.2-I2V-A14B-Diffusers \
  --image-path assets/logo.png \
  --dit-layerwise-offload false --text-encoder-cpu-offload false \
  --vae-cpu-offload false --pin-cpu-memory --dit-cpu-offload false \
  --prompt 'A curious raccoon' --save-output --num-gpus 1 --num-frames 81

echo '=== I2V 4-GPU (93 frames, TP=4) ==='
sglang generate --model-path Wan-AI/Wan2.2-I2V-A14B-Diffusers \
  --image-path assets/logo.png \
  --dit-layerwise-offload false --text-encoder-cpu-offload false \
  --vae-cpu-offload false --pin-cpu-memory --dit-cpu-offload false \
  --prompt 'A curious raccoon' --save-output --num-gpus 4 --tp-size 4 --num-frames 93

echo '=== I2V 8-GPU (93 frames, TP=8) ==='
sglang generate --model-path Wan-AI/Wan2.2-I2V-A14B-Diffusers \
  --image-path assets/logo.png \
  --dit-layerwise-offload false --text-encoder-cpu-offload false \
  --vae-cpu-offload false --pin-cpu-memory --dit-cpu-offload false \
  --prompt 'A curious raccoon' --save-output --num-gpus 8 --tp-size 8 --num-frames 93

echo '=== ALL I2V BENCHMARKS COMPLETE ==='
SCRIPT
chmod +x /scratch/i2v_benchmarks.sh

sudo docker run -d --name i2v-bench --gpus all \
  -v /scratch:/scratch -v /scratch/cache:/root/.cache --ipc=host \
  ${DOCKER_IMAGE} \
  /bin/bash -c '/scratch/i2v_benchmarks.sh 2>&1 | tee /scratch/i2v_output.log'

echo 'I2V benchmarks started in container.'
" 2>&1 | tee -a "${RESULTS_DIR}/i2v_start.log"

# ================================================================
# Step 7: Wait for completion and collect results
# ================================================================
echo "[Step 7/7] Waiting for benchmarks to complete (this takes ~2 hours)..."
echo "  Monitor with:"
echo "    ${SSH_CMD} ${VM_T2V} --command='sudo docker logs --tail 5 t2v-bench 2>&1'"
echo "    ${SSH_CMD} ${VM_I2V} --command='sudo docker logs --tail 5 i2v-bench 2>&1'"
echo ""

# Poll for completion
while true; do
  T2V_STATUS=$(${SSH_CMD} ${VM_T2V} --command="sudo docker ps -a --format '{{.Status}}' --filter name=t2v-bench" 2>/dev/null | grep -c "Exited" || true)
  I2V_STATUS=$(${SSH_CMD} ${VM_I2V} --command="sudo docker ps -a --format '{{.Status}}' --filter name=i2v-bench" 2>/dev/null | grep -c "Exited" || true)

  echo "  [$(date +%H:%M)] T2V: $([ "$T2V_STATUS" -gt 0 ] && echo 'DONE' || echo 'running') | I2V: $([ "$I2V_STATUS" -gt 0 ] && echo 'DONE' || echo 'running')"

  if [ "$T2V_STATUS" -gt 0 ] && [ "$I2V_STATUS" -gt 0 ]; then
    break
  fi
  sleep 300  # Check every 5 minutes
done

echo ""
echo "All benchmarks complete! Collecting results..."

# Collect full logs
${SSH_CMD} ${VM_T2V} --command="sudo docker logs t2v-bench 2>&1" > "${RESULTS_DIR}/t2v_full.log" 2>&1
${SSH_CMD} ${VM_I2V} --command="sudo docker logs i2v-bench 2>&1" > "${RESULTS_DIR}/i2v_full.log" 2>&1

# Extract key metrics
echo ""
echo "============================================="
echo "  RESULTS SUMMARY"
echo "============================================="
echo ""
echo "=== T2V Results ==="
grep -E "average time per step|Pixel data generated|Output saved|CUDA out of memory|Generated [0-9]" "${RESULTS_DIR}/t2v_full.log" 2>/dev/null || echo "  (check t2v_full.log)"
echo ""
echo "=== I2V Results ==="
grep -E "average time per step|Pixel data generated|Output saved|CUDA out of memory|Generated [0-9]" "${RESULTS_DIR}/i2v_full.log" 2>/dev/null || echo "  (check i2v_full.log)"
echo ""
echo "Full logs saved to: ${RESULTS_DIR}/"
echo ""

# ================================================================
# Cleanup prompt
# ================================================================
echo "============================================="
echo "  CLEANUP"
echo "============================================="
echo ""
echo "To delete VMs and stop charges, run:"
echo "  gcloud compute instances delete ${VM_T2V} ${VM_I2V} \\"
echo "    --zone=${ZONE} --project=${PROJECT_ID} --quiet --delete-disks=all"
echo ""
echo "Done!"
