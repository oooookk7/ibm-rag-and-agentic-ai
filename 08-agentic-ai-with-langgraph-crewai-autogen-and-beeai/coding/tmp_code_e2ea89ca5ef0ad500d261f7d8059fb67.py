import numpy as np
import matplotlib.pyplot as plt
import os

# Generate x values from -2π to 2π
x = np.linspace(-2 * np.pi, 2 * np.pi, 1000)

# Calculate sine values
y = np.sin(x)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', linewidth=2, label='sin(x)')
plt.xlabel('x (radians)')
plt.ylabel('sin(x)')
plt.title('Sine Wave from -2π to 2π')
plt.grid(True, alpha=0.3)
plt.xticks(
    [-2*np.pi, -np.pi, 0, np.pi, 2*np.pi],
    ['-2π', '-π', '0', 'π', '2π']
)
plt.legend()
plt.tight_layout()

# Save the plot
plt.savefig('sine_wave.png', dpi=300, bbox_inches='tight')

# Verify file was created
if os.path.exists('sine_wave.png'):
    print("SUCCESS: Plot saved as 'sine_wave.png'")
    print(f"File size: {os.path.getsize('sine_wave.png')} bytes")
else:
    print("ERROR: File was not created")

plt.close()