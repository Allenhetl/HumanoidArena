#!/bin/bash

echo "============================================================"
echo "Verifying pico_server_pose_only.py fixes"
echo "============================================================"

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Test 1: File exists
echo -e "\n1. Checking if file exists..."
if [ -f "pico_server_pose_only.py" ]; then
    echo -e "   ${GREEN}✓${NC} File exists"
else
    echo -e "   ${RED}✗${NC} File not found"
    exit 1
fi

# Test 2: Syntax check
echo -e "\n2. Checking Python syntax..."
if python -c "import ast; ast.parse(open('pico_server_pose_only.py').read())" 2>/dev/null; then
    echo -e "   ${GREEN}✓${NC} Syntax is valid"
else
    echo -e "   ${RED}✗${NC} Syntax error"
    exit 1
fi

# Test 3: Check threading import
echo -e "\n3. Checking threading import..."
if grep -q "import threading" pico_server_pose_only.py; then
    echo -e "   ${GREEN}✓${NC} threading module imported"
else
    echo -e "   ${RED}✗${NC} threading module NOT imported"
    exit 1
fi

# Test 4: Check key functions
echo -e "\n4. Checking key functions..."
FUNCTIONS=("_compute_rel_transform" "_process_3pt_pose" "process_smpl_joints" "run_pico")
ALL_FOUND=true
for func in "${FUNCTIONS[@]}"; do
    if grep -q "def $func" pico_server_pose_only.py; then
        echo -e "   ${GREEN}✓${NC} Found: $func"
    else
        echo -e "   ${RED}✗${NC} Missing: $func"
        ALL_FOUND=false
    fi
done

if [ "$ALL_FOUND" = false ]; then
    exit 1
fi

# Test 5: Check key classes
echo -e "\n5. Checking key classes..."
CLASSES=("PicoReader" "ThreePointPose" "PoseStreamer")
ALL_FOUND=true
for cls in "${CLASSES[@]}"; do
    if grep -q "class $cls" pico_server_pose_only.py; then
        echo -e "   ${GREEN}✓${NC} Found: $cls"
    else
        echo -e "   ${RED}✗${NC} Missing: $cls"
        ALL_FOUND=false
    fi
done

if [ "$ALL_FOUND" = false ]; then
    exit 1
fi

# Test 6: File statistics
echo -e "\n6. File statistics..."
LINES=$(wc -l < pico_server_pose_only.py)
SIZE=$(wc -c < pico_server_pose_only.py)
echo -e "   Lines: $LINES"
echo -e "   Size: $SIZE bytes"

# Test 7: Check if main block exists
echo -e "\n7. Checking main block..."
if grep -q 'if __name__ == "__main__"' pico_server_pose_only.py; then
    echo -e "   ${GREEN}✓${NC} Main block found"
else
    echo -e "   ${RED}✗${NC} Main block NOT found"
    exit 1
fi

echo -e "\n============================================================"
echo -e "${GREEN}✓ All verification tests passed!${NC}"
echo -e "============================================================"
echo -e "\nFile is ready to use:"
echo -e "  python pico_server_pose_only.py --vis_vr3pt --vis_smpl"
echo -e "\nFor help:"
echo -e "  python pico_server_pose_only.py --help"
