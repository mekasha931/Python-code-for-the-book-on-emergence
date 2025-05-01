import numpy as np
import matplotlib.pyplot as plt

N = 50  # spatial sites
T = 100  # time steps
diffusion_rate = 0.1
remodel_rate = 0.05
feedback_strength = 0.2

x = np.random.rand(N) * 0.1  # initial ECM density
X = []

for t in range(T):
    # Macro property: average ECM density
    X_t = np.mean(x)
    X.append(X_t)
    
    x_new = np.copy(x)
    for i in range(N):
        # Diffusion with neighbors
        left = x[i-1] if i > 0 else x[i]
        right = x[i+1] if i < N-1 else x[i]
        diffusion = diffusion_rate * (left + right - 2*x[i])
        
        # Remodeling influenced by macro property (feedback)
        remodeling = remodel_rate * X_t * (1 - x[i])
        
        x_new[i] += diffusion + remodeling
        x_new[i] = min(max(x_new[i], 0), 1)  # clamp between 0 and 1
    x = x_new

plt.plot(X)
plt.xlabel('Time')
plt.ylabel('Average ECM density')
plt.title('Emergence of Tissue Structure via ECM Remodeling')
plt.show()
