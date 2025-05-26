import unittest
import numpy as np
import cuda_ceos_py as ceos
import src.analysis_utils as au


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

    def test_msong(self):
        X, Q = au.write_load_datasets.load_dataset("msong")
        D = 1024
        m = 100
        s_0 = 1

        index, R = ceos.indexing_hybridCEOs(X, D, m, s_0)

        k = 10
        b = 50
        s = 100

        top_indices, distances = ceos.querying_hybridCEOs(index, X, R, Q, k, D, s, b, use_faiss_top_k=True)

        # TODO, add code here to save the index to disk and load it back, instead of re-indexing

        top_indices_exact, distances_exact = au.perform_exact_nns(X, Q, k)
        recall_value = au.recall(top_indices, top_indices_exact, k)

        print(f"Recall: {recall_value:.4f}")
