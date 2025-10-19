# import math, time

# x = list(range(-400, 400, 2))

# def sigmoid(x):
#     ans = 0
#     final_result = []
    
#     for i in range(len(x)):
#         start_time = time.time()
#         ans = (1 / (1 + math.exp(-x[i])))
#         final_result.append(ans)
#     end_time = time.time()
#     print(f"Total time: {end_time - start_time:.8f} seconds")
#     return final_result

# print(sigmoid(x))

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.linspace(-10, 10, 100)
y = sigmoid(x)

plt.plot(x, y)
plt.title("Sigmoid Function")
plt.xlabel("x")
plt.ylabel("sigmoid(x)")
plt.grid(True)
plt.show()
