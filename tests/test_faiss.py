import src.analysis_utils as au
# import unittest
import faiss
import time 
import numpy as np

dataset_name = "msong"
X, Q = au.write_load_datasets.load_dataset(dataset_name)

d = X.shape[1]
k = 10
nlist = 512  # number of coarse clusters
nprobe = 6  # number of clusters to probe during search

# GPU resources
res = faiss.StandardGpuResources()

# Build approximate MIPS index on GPU
quantizer = faiss.GpuIndexFlatIP(res, d)
gpu_index = faiss.GpuIndexIVFFlat(res, quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)

gpu_index.train(X)
gpu_index.add(X)
gpu_index.nprobe = nprobe

# Warm-up
gpu_index.search(Q, k)

# Timed runs
times = []
for _ in range(10):
    start = time.perf_counter()
    _, top_indices = gpu_index.search(Q, k)
    end = time.perf_counter()
    times.append(end - start)

avg_time = np.mean(times)
print(f"Average FAISS search time over 10 runs: {avg_time:.6f} seconds")

# top_indices_exact, _ = au.perform_exact_nns(X, Q, k)

# recall_value = au.recall(top_indices, top_indices_exact, k)

# print(f"Recall: {recall_value:.4f}")

index = faiss.IndexFlatIP(X.shape[1])  # for inner product (dot product)
index.add(X)
_, top_indices_exact = index.search(Q, 10)

recall_value = au.recall(top_indices, top_indices_exact, k)

print(f"Recall: {recall_value:.4f}")