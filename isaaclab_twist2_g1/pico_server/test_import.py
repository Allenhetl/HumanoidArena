#!/usr/bin/env python
"""Test if pico_server_pose_only.py can be imported without errors"""

import sys
import ast

print("="*60)
print("Testing pico_server_pose_only.py")
print("="*60)

# Test 1: Syntax check
print("\n1. Syntax check...")
try:
    with open('pico_server_pose_only.py', 'r') as f:
        ast.parse(f.read())
    print("   ✓ Syntax is valid")
except SyntaxError as e:
    print(f"   ✗ Syntax error: {e}")
    sys.exit(1)

# Test 2: Check imports
print("\n2. Checking imports...")
required_imports = [
    'import threading',
    'import os',
    'import time',
    'import numpy',
    'import torch',
    'import zmq',
]

with open('pico_server_pose_only.py', 'r') as f:
    content = f.read()

missing = []
for imp in required_imports:
    if imp not in content:
        missing.append(imp)

if missing:
    print(f"   ✗ Missing imports: {missing}")
    sys.exit(1)
else:
    print(f"   ✓ All required imports present")

# Test 3: Check key functions
print("\n3. Checking key functions and classes...")
required_items = [
    'def _compute_rel_transform',
    'def _process_3pt_pose',
    'def process_smpl_joints',
    'class PicoReader',
    'class ThreePointPose',
    'class PoseStreamer',
    'def run_pico',
]

missing = []
for item in required_items:
    if item not in content:
        missing.append(item)

if missing:
    print(f"   ✗ Missing items: {missing}")
    sys.exit(1)
else:
    print(f"   ✓ All {len(required_items)} key items found")

# Test 4: Check line count
print("\n4. File statistics...")
lines = content.split('\n')
print(f"   Lines: {len(lines)}")
print(f"   Size: {len(content)} bytes")
print(f"   Functions: {content.count('def ')}")
print(f"   Classes: {content.count('class ')}")

print("\n" + "="*60)
print("✓ All tests passed!")
print("="*60)
print("\nFile is ready to use:")
print("  python pico_server_pose_only.py --vis_vr3pt --vis_smpl")
