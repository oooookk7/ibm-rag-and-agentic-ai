import sys
print(f"Python version: {sys.version}")

try:
    import numpy as np
    print("✓ numpy is available")
except ImportError:
    print("✗ numpy is NOT available")

try:
    import matplotlib.pyplot as plt
    print("✓ matplotlib is available")
except ImportError:
    print("✗ matplotlib is NOT available")