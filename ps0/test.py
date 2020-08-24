import numpy as np

matrix = np.random.rand(100, 100) * 10

print(matrix)

np.save("q3-input.npy", matrix)