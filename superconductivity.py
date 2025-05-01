import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 50  # number of sites
T = 100  # time steps
alpha = 0.1  # local growth rate
beta = 0.05  # nonlinear saturation
interaction_range = 5  # range of interaction
interaction_strength = 0.2

# Initialize electron pair densities randomly
x = np.random.rand(N)
X = []

def kernel(i, j):
    # Interaction decays exponentially with distance
    dist = abs(i - j)
    return interaction_strength * np.exp(-dist / interaction_range)

for t in range(T):
    # Compute macro order parameter as average density
    X_t = np.mean(x)
    X.append(X_t)
    
    x_new = np.zeros_like(x)
    for i in range(N):
        # Local dynamics with nonlinear saturation
        local_growth = alpha * x[i] * (1 - beta * x[i])
        
        # Interaction term: sum over neighbors weighted by kernel
        interaction = sum(kernel(i, j) * x[j] for j in range(N))
        
        # Feedback from macro order parameter
        feedback = 0.1 * X_t
        
        # Update rule
        x_new[i] = x[i] + local_growth + interaction + feedback
        # Ensure positivity
        x_new[i] = max(0, x_new[i])
    x = x_new

# Plot evolution of macro order parameter
plt.plot(X)
plt.xlabel('Time')
plt.ylabel('Superconducting order parameter')
plt.title('Emergence of Superconductivity')
plt.show()
