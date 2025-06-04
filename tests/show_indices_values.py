import numpy as np
import matplotlib.pyplot as plt

# Paste both arrays and concatenate them
index = np.load(r"/workspace/CUDA-CEOs/CUDA-CEOs-py/tests/saved_indices/gist_index_shuffle_D1024_m100_s01.npy")

values = index.flatten()

# Plot histogram
plt.figure(figsize=(10, 5))
plt.hist(values, bins=50, color='steelblue', edgecolor='black')
plt.title("Histogram of Values")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()
