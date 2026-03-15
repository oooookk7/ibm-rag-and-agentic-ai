import numpy as np
import matplotlib.pyplot as plt

# Generate x values from -2π to 2π
x = np.linspace(-2 * np.pi, 2 * np.pi, 1000)

# Calculate sine values
y = np.sin(x)

# Create and configure the plot
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', linewidth=2)
plt.xlabel('x (radians)')
plt.ylabel('sin(x)')
plt.title('Sine Wave from -2π to 2π')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the plot
plt.savefig('sine_wave.png', dpi=300)
plt.show()