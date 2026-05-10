#!/bin/bash
# AARS — Dynamic Song Processor
# Usage: ./infra/process_songs.sh /path/to/my/songs

set -e

SONGS_PATH=$1

if [ -z "$SONGS_PATH" ]; then
    echo "Usage: $0 /path/to/my/songs"
    exit 1
fi

# Convert to absolute path if it's relative
ABS_SONGS_PATH=$(cd "$(dirname "$SONGS_PATH")" && pwd)/$(basename "$SONGS_PATH")

echo "╔════════════════════════════════════════════╗"
echo "║  AARS — Dynamic Processing Session         ║"
echo "╚════════════════════════════════════════════╝"
echo "Target Path: $ABS_SONGS_PATH"
echo ""

# 1. Ensure the directory exists
if [ ! -d "$ABS_SONGS_PATH" ]; then
    echo "[!] Error: Directory $ABS_SONGS_PATH does not exist."
    exit 1
fi

# 2. Run the pipeline with the custom path
# We mount the user's path to /data/input inside the container
# We also use the --purge flag to ensure a clean start per section
echo "[*] Launching AURA-DSP Swarm..."
AURA_DATA_PATH="$ABS_SONGS_PATH" docker-compose run --rm aura-pipeline python pipeline/main.py --input /data --purge

echo ""
echo "╔════════════════════════════════════════════╗"
echo "║  PROCESSING SESSION COMPLETE               ║"
echo "╚════════════════════════════════════════════╝"
