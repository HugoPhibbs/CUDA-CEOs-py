import numpy as np
import matplotlib.pyplot as plt

# Load raw binary vector (assume float32 and known shape)
data = np.fromfile("/workspace/CUDA-CEOs/histogram.bin", dtype=np.int32)

# Reshape if needed — e.g. (n_rows, n_cols)
tensor_np = data.reshape((20, -1))  # adjust 256 to match your C++ shape

# Assume tensor_np has shape (20, 1_000_000)
usable_cols = 994000
trimmed = tensor_np[:, :usable_cols]  # shape: (20, 994000)

# Reshape and average
reduced = trimmed.reshape(20, 1988, 500).mean(axis=2)  # shape: (20, 1988)

# # Get top-50 indices per row after sorting descending
# topk_indices = np.argsort(-tensor_np, axis=1)[:, :50]  # shape: (n_rows, 50)

# # Display heatmap
plt.figure(figsize=(12, 6))
plt.imshow(reduced, aspect='auto', interpolation='nearest', cmap='inferno')
plt.colorbar(label='Index Value')
plt.title("Top-50 Sorted Indices per Row")
plt.xlabel("Rank")
plt.ylabel("Row Index")
plt.show()
