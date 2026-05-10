#!/bin/bash
echo "Waiting for Qwen2-Audio on port 8001..."
for i in $(seq 1 30); do
    if curl -s http://localhost:8001/v1/models 2>/dev/null | grep -q Qwen; then
        echo "QWEN2_AUDIO_ONLINE"
        curl -s http://localhost:8001/v1/models
        exit 0
    fi
    echo "  Attempt $i/30 - still loading..."
    sleep 5
done
echo "TIMEOUT after 150s"
echo "=== Container Status ==="
docker ps --filter name=vllm-qwen2-audio --format 'table {{.Names}}\t{{.Status}}'
echo "=== Last 20 log lines ==="
docker logs --tail 20 vllm-qwen2-audio 2>&1
exit 1
