import numpy as np
import matplotlib.pyplot as plt

# Generate x values from -2π to 2π
x = np.linspace(-2 * np.pi, 2 * np.pi, 1000)

# Calculate corresponding y values (sine of x)
y = np.sin(x)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', linewidth=2, label='sin(x)')

# Add labels and title
plt.xlabel('x (radians)', fontsize=12)
plt.ylabel('sin(x)', fontsize=12)
plt.title('Sine Wave from -2π to 2π', fontsize=14, fontweight='bold')

# Add grid for better readability
plt.grid(True, alpha=0.3)

# Add x-axis ticks at multiples of π
plt.xticks(
    [-2*np.pi, -1.5*np.pi, -np.pi, -0.5*np.pi, 0, 0.5*np.pi, np.pi, 1.5*np.pi, 2*np.pi],
    ['-2π', '-3π/2', '-π', '-π/2', '0', 'π/2', 'π', '3π/2', '2π']
)

# Add legend
plt.legend()

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot
plt.savefig('sine_wave.png', dpi=300, bbox_inches='tight')

# Display the plot
plt.show()

print("Plot saved as 'sine_wave.png'")