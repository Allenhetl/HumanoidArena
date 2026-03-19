#!/usr/bin/env python3
"""测试 matplotlib 不带工具栏"""

import matplotlib
matplotlib.rcParams['toolbar'] = 'None'  # Disable toolbar

try:
    matplotlib.use('Qt5Agg')
    print("Using Qt5Agg")
except:
    try:
        matplotlib.use('TkAgg')
        print("Using TkAgg")
    except:
        print("Using default backend")

import matplotlib.pyplot as plt
import numpy as np

print(f"Backend: {matplotlib.get_backend()}")

fig, ax = plt.subplots(figsize=(10, 6))
x = np.linspace(0, 10, 100)
ax.plot(x, np.sin(x))
ax.set_title("Test Plot - No Toolbar")
print("Showing plot...")
plt.show()
print("Success!")
