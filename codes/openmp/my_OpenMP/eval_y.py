import numpy as np

# Parameters
n = 100_000
dx = 1.0 / n

# Create x array from 1*dx to n*dx
x = np.arange(1, n + 1) * dx

# Compute y = sin(x) * exp(-x^2)
y = np.sin(x) * np.exp(-x**2)

print(f"y array has been filled with n = {n}")

