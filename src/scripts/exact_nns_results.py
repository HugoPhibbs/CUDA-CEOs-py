import faiss
import numpy as np
import time

print(faiss.get_num_gpus())


res = faiss.StandardGpuResources()

X_gist = np.fromfile("/workspace/CUDA-CEOs/datasets/bin/gist/gist_dataset.bin", dtype=np.float32)
Q_gist = np.fromfile("/workspace/CUDA-CEOs/datasets/bin/gist/gist_queries.bin", dtype=np.float32)

d = 960

X_gist = X_gist.reshape(-1, d)
Q_gist = Q_gist.reshape(-1, d)

index = faiss.IndexFlatL2(d)
index_gpu = faiss.index_cpu_to_gpu(res, 0, index)

start = time.time()
index_gpu.add(X_gist)
end = time.time()
print("Time taken to add vectors:", end - start)

print(index.is_trained)

