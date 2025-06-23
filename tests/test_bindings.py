import unittest
import numpy as np
import torch
import cuda_ceos_py as ceos
import src.analysis_utils as au
import os
import time

class coCEOsTest(unittest.TestCase):

    def test_basic(self):
        n = 10_000
        d = 30
        D = 12
        m = 50
        k = 1
        b = 2
        s_0 = 2
        n_queries = 50
        
        X = np.random.normal(size=(n, d)).astype(np.float32)

        Q = np.random.normal(size=(n_queries, d)).astype(np.float32)

        try:
            L, S, R = ceos.indexing_coCEOs(X, D, m)
            top_indices, distances = ceos.querying_coCEOs(L, S, X, R, Q, b, k, s_0, use_faiss_top_k=True) 

            top_indices_exact, distances_exact = au.perform_exact_nns(X, Q, k)

            recall_value = au.recall(top_indices, top_indices_exact, k)
            print(f"Recall: {recall_value:.4f}")
        except:
            self.fail("coCEOs indexing raised an exception unexpectedly!")

    def test_gist(self):
        X, Q = au.write_load_datasets.load_dataset("gist")
        D = 1024
        m = 100

        index, R = ceos.indexing_coCEOs(X, D, m)

        k = 10
        b = 50
        s_0 = 100

        top_indices, distances = ceos.querying_coCEOs(index, X, R, Q, b, k, s_0, use_faiss_top_k=True)

        top_indices_exact, distances_exact = au.perform_exact_nns(X, Q, k)
        recall_value = au.recall(top_indices, top_indices_exact, k)

        print(f"Recall: {recall_value:.4f}")


class hybridCEOsTest(unittest.TestCase):

    def test_for_real_dataset(self):
        dataset_name = "msong"
        X, Q = au.write_load_datasets.load_dataset(dataset_name)
        D = 1024
        m = 10
        s_0 = 1
        
        X_torch = torch.from_numpy(X).cuda()
        Q_torch = torch.from_numpy(Q).cuda()

        hybrid_ceos = ceos.HybridCEOs(X_torch, D, m, s_0)

        hybrid_ceos.set_use_low_memory_hist(False)

        index_path = f"/workspace/CUDA-CEOs/CUDA-CEOs-py/tests/saved_indices/{dataset_name}_index_D{D}_m{m}_s0{s_0}.pt"
        index_sums_path = f"/workspace/CUDA-CEOs/CUDA-CEOs-py/tests/saved_indices/{dataset_name}_index_sums_D{D}_m{m}_s0{s_0}.pt"
        R_path = f"/workspace/CUDA-CEOs/CUDA-CEOs-py/tests/saved_indices/{dataset_name}_R_D{D}_m{m}_s0{s_0}.pt"

        if os.path.exists(index_path):
            index = torch.load(index_path).cuda()
            index_sums = torch.load(index_sums_path).cuda()
            R = torch.load(R_path).cuda()

            hybrid_ceos.load_index(index, index_sums, R)
        else:
            index, index_sums, R = hybrid_ceos.index()

            torch.save(index, index_path)
            torch.save(index_sums, index_sums_path)
            torch.save(R, R_path)

        k = 10
        b = 50
        s = 100

        top_indices, distances = hybrid_ceos.query_s01(Q_torch, k, s, b)

        # Averaged timing over N trials
        trials = 10
        total_ms = 0.0

        for _ in range(trials):
            # R_new = np.random.normal(size=(D, X.shape[1])).astype(np.float32)  # Simulate a new R matrix
            start = time.time()
            _ = hybrid_ceos.query_s01(Q_torch, k, s, b)
            end = time.time()
            total_ms += (end - start) * 1000  # milliseconds

        print(f"Avg querying time over {trials} trials: {total_ms / trials:.3f} ms")

        exact_indices_path = f"/workspace/CUDA-CEOs/CUDA-CEOs-py/tests/saved_results/{dataset_name}_exact_indices_k{k}.npy"
        exact_distances_path = f"/workspace/CUDA-CEOs/CUDA-CEOs-py/tests/saved_results/{dataset_name}_exact_distances_k{k}.npy"

        if os.path.exists(exact_indices_path) and os.path.exists(exact_distances_path):
            top_indices_exact = np.load(exact_indices_path)
            distances_exact = np.load(exact_distances_path)
        else:
            top_indices_exact, distances_exact = au.perform_exact_nns(X, Q, k)
            np.save(exact_indices_path, top_indices_exact)
            np.save(exact_distances_path, distances_exact)

        recall_value = au.recall(top_indices.cpu().numpy(), top_indices_exact, k)

        print(f"Recall: {recall_value:.4f}")
        