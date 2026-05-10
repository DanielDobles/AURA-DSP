#!/bin/bash
# AARS — vLLM Deployment (ROCm MI300X)
# Multi-model strategy using Qwen family:
#   1. Qwen3-32B (or 235B-A22B MoE) → Agent orchestration & function calling
#   2. Qwen2-Audio-7B-Instruct → Multimodal audio understanding ("listening")
#
# With 192GB HBM3, both models fit comfortably.

set -e

PORT_LLM=8000
PORT_AUDIO=8001
CONTAINER_LLM="vllm-qwen3"
CONTAINER_AUDIO="vllm-qwen2-audio"

# ─── Model Selection ──────────────────────────────────
# Qwen3 for agent brain (function calling, reasoning)
# Options by VRAM budget:
#   - Qwen/Qwen3-235B-A22B  → MoE, ~50GB active, best quality
#   - Qwen/Qwen3-32B        → Dense, ~65GB FP16, solid balance
#   - Qwen/Qwen3-14B        → Dense, ~28GB FP16, fast & cheap
MODEL_LLM="Qwen/Qwen3-32B"

# Qwen2-Audio for audio understanding (the agent that "listens")
MODEL_AUDIO="Qwen/Qwen2-Audio-7B-Instruct"

echo "╔════════════════════════════════════════════╗"
echo "║  vLLM Multi-Model Deploy (ROCm MI300X)    ║"
echo "╚════════════════════════════════════════════╝"

# ─── Aggressive Cleanup (purge ALL vLLM/old containers) ───
echo "[*] Purging ALL existing vLLM and ROCm containers..."
# Kill by name patterns (AURYGA leftovers + our own)
for name in vllm-reasoning vllm-coder vllm-qwen3 vllm-qwen2-audio rocm; do
    docker rm -f $name 2>/dev/null && echo "  Removed: $name" || true
done
# Kill any remaining vllm containers by image
docker ps -a --filter "ancestor=vllm/vllm-openai-rocm:latest" -q | xargs -r docker rm -f 2>/dev/null || true
docker ps -a --filter "ancestor=vllm/vllm-omni-rocm:latest" -q | xargs -r docker rm -f 2>/dev/null || true
docker ps -a --filter "ancestor=rocm/vllm:latest" -q | xargs -r docker rm -f 2>/dev/null || true
echo "[+] All old containers purged."

echo "[*] Pulling vLLM ROCm image..."
docker pull vllm/vllm-openai-rocm:latest

# ─── Deploy Qwen3 (Agent Brain) ──────────────────────
echo ""
echo "▶ Deploying ${MODEL_LLM} on port ${PORT_LLM}..."
docker run -d \
    --name $CONTAINER_LLM \
    --device /dev/kfd \
    --device /dev/dri \
    --group-add video \
    --ipc=host \
    --shm-size 16g \
    -p 127.0.0.1:${PORT_LLM}:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -e HSA_OVERRIDE_GFX_VERSION=9.4.2 \
    -e HIP_VISIBLE_DEVICES=0 \
    vllm/vllm-openai-rocm:latest \
    --model $MODEL_LLM \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.75 \
    --trust-remote-code \
    --dtype auto \
    --enable-auto-tool-choice \
    --tool-call-parser hermes

# ─── Deploy Qwen2-Audio (Listener) ───────────────────
echo ""
echo "▶ Deploying ${MODEL_AUDIO} on port ${PORT_AUDIO}..."
docker run -d \
    --name $CONTAINER_AUDIO \
    --device /dev/kfd \
    --device /dev/dri \
    --group-add video \
    --ipc=host \
    --shm-size 8g \
    -p 127.0.0.1:${PORT_AUDIO}:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -e HSA_OVERRIDE_GFX_VERSION=9.4.2 \
    -e HIP_VISIBLE_DEVICES=0 \
    vllm/vllm-openai-rocm:latest \
    --model $MODEL_AUDIO \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.15 \
    --trust-remote-code \
    --dtype auto

# ─── Health Check ─────────────────────────────────────
echo ""
echo "[*] Waiting for models to load..."
for port in $PORT_LLM $PORT_AUDIO; do
    for i in $(seq 1 60); do
        if curl -s http://localhost:${port}/v1/models > /dev/null 2>&1; then
            echo "[+] Port ${port} is READY!"
            curl -s http://localhost:${port}/v1/models | python3 -m json.tool
            break
        fi
        echo "  Waiting port ${port}... ($i/60)"
        sleep 5
    done
done

echo ""
echo "╔════════════════════════════════════════════╗"
echo "║  MODELS DEPLOYED                          ║"
echo "║  Qwen3 (brain):  http://localhost:${PORT_LLM}   ║"
echo "║  Qwen2-Audio:    http://localhost:${PORT_AUDIO}   ║"
echo "╚════════════════════════════════════════════╝"
echo ""
echo "VRAM Budget (192GB):"
echo "  Qwen3-32B FP16:     ~65GB (40%)"
echo "  Qwen2-Audio-7B:     ~29GB (15%)"
echo "  Audio ML models:    ~50GB (remaining)"
echo "  System/KV cache:    ~48GB (buffer)"
