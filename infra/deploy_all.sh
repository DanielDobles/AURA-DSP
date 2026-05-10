#!/bin/bash
# AARS — Master Deploy Script
# Run this ONCE when the Droplet boots. Does everything.
# Usage: ssh root@IP 'bash -s' < infra/deploy_all.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "╔════════════════════════════════════════════╗"
echo "║  AARS — Full Deployment Pipeline           ║"
echo "║  $(date)                     ║"
echo "╚════════════════════════════════════════════╝"

# Step 1: Server setup
echo ""
echo "▶ PHASE 1: Server Hardening..."
bash /tmp/aars/infra/setup_server.sh

# Step 2: vLLM
echo ""
echo "▶ PHASE 2: vLLM Deployment..."
bash /tmp/aars/infra/deploy_vllm.sh

# Step 3: Validate GPU
echo ""
echo "▶ PHASE 3: GPU Validation..."
rocm-smi
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'ROCm available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NONE\"}')" 2>/dev/null || echo "[!] PyTorch not installed on host (expected — it runs in Docker)"

echo ""
echo "╔════════════════════════════════════════════╗"
echo "║  DEPLOYMENT COMPLETE                       ║"
echo "╚════════════════════════════════════════════╝"
echo ""
echo "Verify with:"
echo "  curl http://localhost:8000/v1/models"
echo "  rocm-smi"
echo "  docker ps"
echo ""
echo "To process songs with a dynamic path:"
echo "  bash infra/process_songs.sh /path/to/your/songs"
