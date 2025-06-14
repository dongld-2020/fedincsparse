import numpy as np
import matplotlib.pyplot as plt

initial_retention = 0.1
final_retention = 0.9
growth_rate = 0.1
rounds = np.arange(1, 101)

retention_factors = initial_retention + (final_retention - initial_retention) * (1 - np.exp(-growth_rate * rounds))

# Nếu bạn muốn in 1 vài giá trị đầu:
for i in range(10):
    print(f"Round {i+1}: retention_factor = {retention_factors[i]:.4f}")

# Hoặc vẽ đồ thị:
plt.plot(rounds, retention_factors)
plt.xlabel('Round Number')
plt.ylabel('Retention Factor')
plt.title('Retention Factor Over Rounds')
plt.grid(True)
plt.show()
