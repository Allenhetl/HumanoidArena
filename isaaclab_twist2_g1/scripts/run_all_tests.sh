#!/bin/bash
# Copyright (c) 2025, HumanoidArena Project
# Run all terrain, visual zone, and football environment tests
# Uses conda environment: HumanoidArena-xzk

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Activate conda environment HumanoidArena-xzk
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi
if command -v conda &>/dev/null; then
    conda activate HumanoidArena-xzk
    echo "Using conda env: $CONDA_DEFAULT_ENV"
fi

echo "============================================================"
echo "HumanoidArena Test Suite"
echo "============================================================"

# Test 1: Terrain generation (no IsaacLab required)
echo ""
echo "[Test 1/4] Testing terrain generation functions..."
echo "------------------------------------------------------------"
python scripts/test_terrain_generation.py

# Test 2: Terrain environments (requires IsaacLab)
echo ""
echo "[Test 2/4] Testing terrain environments..."
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
echo "[Test 3/4] Testing visual zones environment..."
echo "------------------------------------------------------------"
python scripts/test_visual_zones_env.py \
    --headless \
    --device cuda \
    --num_steps 50

# Test 4: Football environment (G1 29DOF Dex3 + football)
echo ""
echo "[Test 4/4] Testing football environment..."
echo "------------------------------------------------------------"
python scripts/test_football_env.py \
    --headless \
    --device cuda \
    --num_steps 50

echo ""
echo "============================================================"
echo "All tests completed successfully!"
echo "============================================================"
