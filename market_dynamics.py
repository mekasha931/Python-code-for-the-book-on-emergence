import numpy as np
import matplotlib.pyplot as plt

N = 200  # agents
T = 100  # time steps
imitation_strength = 0.4
noise_level = 0.1

# Initial propensities [-1 sell, 1 buy]
x = np.random.uniform(-1, 1, N)
prices = []

for t in range(T):
    # Macro price is weighted average of propensities
    X_t = np.tanh(np.mean(x))
    prices.append(X_t)
    
    x_new = np.copy(x)
    for i in range(N):
        # Imitate average propensity plus noise
        local_influence = np.mean(x)
        x_new[i] = (1 - imitation_strength) * x[i] + imitation_strength * local_influence + np.random.normal(0, noise_level)
        x_new[i] = np.clip(x_new[i], -1, 1)
    x = x_new

plt.plot(prices)
plt.xlabel('Time')
plt.ylabel('Market Price')
plt.title('Emergence of Market Price Dynamics')
plt.show()
