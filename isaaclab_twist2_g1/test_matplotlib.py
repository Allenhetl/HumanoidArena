#!/usr/bin/env python3
"""测试 matplotlib 是否可以显示窗口"""

import sys
print(f"Python: {sys.executable}")

print("\n1. Testing matplotlib import...")
try:
    import matplotlib
    print(f"   matplotlib version: {matplotlib.__version__}")
    print(f"   matplotlib location: {matplotlib.__file__}")
except Exception as e:
    print(f"   ERROR: {e}")
    sys.exit(1)

print("\n2. Testing backend configuration...")
try:
    matplotlib.use('TkAgg')
    print(f"   Backend set to: TkAgg")
except Exception as e:
    print(f"   WARNING: Could not set TkAgg: {e}")

print("\n3. Testing pyplot...")
try:
    import matplotlib.pyplot as plt
    print(f"   pyplot imported successfully")
    print(f"   Current backend: {matplotlib.get_backend()}")
except Exception as e:
    print(f"   ERROR: {e}")
    sys.exit(1)

print("\n4. Testing simple plot...")
try:
    import numpy as np
    fig, ax = plt.subplots()
    x = np.linspace(0, 10, 100)
    ax.plot(x, np.sin(x))
    ax.set_title("Test Plot - If you see this window, matplotlib is working!")
    print("   Plot created, showing window...")
    print("   Close the window to continue...")
    plt.show()
    print("   SUCCESS: matplotlib window displayed!")
except Exception as e:
    print(f"   ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✅ All tests passed! Matplotlib is working correctly.")
