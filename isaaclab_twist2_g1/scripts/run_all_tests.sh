#!/bin/bash
# Copyright (c) 2025, HumanoidArena Project
# Run all terrain and visual zone tests

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "============================================================"
echo "HumanoidArena Terrain & Visual Zones Test Suite"
echo "============================================================"

# Test 1: Terrain generation (no IsaacLab required)
echo ""
echo "[Test 1/3] Testing terrain generation functions..."
echo "------------------------------------------------------------"
python scripts/test_terrain_generation.py

# Test 2: Terrain environments (requires IsaacLab)
echo ""
echo "[Test 2/3] Testing terrain environments..."
echo "------------------------------------------------------------"

TERRAIN_TYPES=("flat" "slope" "stairs" "wave")

for terrain in "${TERRAIN_TYPES[@]}"; do
    echo ""
    echo "Testing terrain: $terrain"
    python scripts/test_terrain_env.py \
        --headless \
        --device cuda \
        --terrain_type "$terrain" \
        --num_steps 50
done

# Test 3: Visual zones environment
echo ""
echo "[Test 3/3] Testing visual zones environment..."
echo "------------------------------------------------------------"
python scripts/test_visual_zones_env.py \
    --headless \
    --device cuda \
    --num_steps 50

echo ""
echo "============================================================"
echo "All tests completed successfully!"
echo "============================================================"
