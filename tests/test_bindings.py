import time
import unittest
import numpy as np
import cuda_ceos_py as ceos
import src.analysis_utils as au
import os


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

        index_path = f"/workspace/CUDA-CEOs/CUDA-CEOs-py/tests/saved_indices/{dataset_name}_index_D{D}_m{m}_s0{s_0}.npy"
        index_sums_path = f"/workspace/CUDA-CEOs/CUDA-CEOs-py/tests/saved_indices/{dataset_name}_index_sums_D{D}_m{m}_s0{s_0}.npy"
        R_path = f"/workspace/CUDA-CEOs/CUDA-CEOs-py/tests/saved_indices/{dataset_name}_R_D{D}_m{m}_s0{s_0}.npy"

        if os.path.exists(index_path):
            index = np.load(index_path)
            index_sums = np.load(index_sums_path)
            R = np.load(R_path)
        else:
            index, index_sums, R = ceos.indexing_hybridCEOs(X, D, m, s_0)
            np.save(index_path, index)
            np.save(index_sums_path, index_sums)
            np.save(R_path, R)

        print(index[0, 1])
        print(index[1, 2])  

        k = 10
        b = 50
        s = 100

        top_indices, distances = ceos.querying_hybridCEOs(index, index_sums, X, R, Q, k, D, s, b, use_faiss_top_k=True)

        print('top indices 0: ', top_indices[0])
        print('top indices 1: ', top_indices[1])

        print('top distances 0: ', distances[0])
        print('top distances 1: ', distances[1])

        # TODO, add code here to save the index to disk and load it back, instead of re-indexing

        _ = ceos.querying_hybridCEOs(index, index_sums, X, R, Q[:10], k, D, s, b, use_faiss_top_k=True)

        # Averaged timing over N trials
        trials = 10
        total_ms = 0.0

        for _ in range(trials):
            R_new = np.random.normal(size=(D, X.shape[1])).astype(np.float32)  # Simulate a new R matrix
            start = time.time()
            _ = ceos.querying_hybridCEOs(index, index_sums, X, R_new, Q, k, D, s, b, use_faiss_top_k=True)
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

        print('top indices exact 0: ', top_indices_exact[0])
        print('top distances exact 0: ', distances_exact[0])

        recall_value = au.recall(top_indices, top_indices_exact, k)

        print(f"Recall: {recall_value:.4f}")

        print(np.dot(Q[0], X[top_indices[0, 5]]))
        print(X[top_indices[0, 5]])
        print(np.dot(Q[0], X[top_indices[0, 6]]))
        print(X[top_indices[0, 6]])
        