import numpy as np
import matplotlib.pyplot as plt

N = 100  # individuals
T = 50  # time steps
influence_strength = 0.3
noise_level = 0.05

# Random initial opinions [-1,1]
x = np.random.uniform(-1, 1, N)
X = []

# Influence kernel based on social network (here simple nearest neighbors)
def kernel(i, j):
    dist = abs(i - j)
    return np.exp(-dist)

for t in range(T):
    X_t = np.mean(x)
    X.append(X_t)
    x_new = np.copy(x)
    for i in range(N):
        influence = sum(kernel(i, j) * x[j] for j in range(N))
        influence /= sum(kernel(i, j) for j in range(N))
        # Update opinion with social influence and noise
        x_new[i] = (1 - influence_strength) * x[i] + influence_strength * influence + np.random.normal(0, noise_level)
        x_new[i] = np.clip(x_new[i], -1, 1)
    x = x_new

plt.plot(X)
plt.xlabel('Time')
plt.ylabel('Average Opinion')
plt.title('Emergence of Social Consensus')
plt.show()
