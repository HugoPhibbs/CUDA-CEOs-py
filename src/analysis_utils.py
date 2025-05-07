import numpy as np
import src.scripts.write_load_datasets as write_load_datasets
import faiss


def recall(actual_indices, expected_indices, k):
    """
    Calculate recall at k.

    Parameters:
    actual_indices (np.ndarray): Actual indices of the nearest neighbors. Shape (n_queries, m), for m>=k. Assume that
                                  actual_indices are sorted. I.e. closest neighbors are at the beginning (column-wise)
    expected_indices (np.ndarray): Expected indices of the nearest neighbors. Shape (n_queries, l), for l>=k. Assume that
                                   expected_indices are sorted. I.e. closest neighbors are at the beginning (column-wise)
    k (int): The number of nearest neighbors to consider.
    """
    assert actual_indices.shape[1] >= k, "actual_indices must have at least k columns"
    assert (
        expected_indices.shape[1] >= k
    ), "expected_indices must have at least k columns"

    actual_top_k = actual_indices[:, :k]
    expected_top_k = expected_indices[:, :k]

    n_queries = actual_top_k.shape[0]

    total = 0

    for i in range(n_queries):
        this_actual_top_k = actual_top_k[i]
        this_expected_top_k = expected_top_k[i]

        intersect = np.intersect1d(this_actual_top_k, this_expected_top_k)

        recall_value = len(intersect) / k

        total += recall_value

    return total / n_queries

def perform_exact_nns(X:np.ndarray, Q:np.ndarray, k:int):
    """
    Perform exact nearest neighbor search using FAISS.

    Parameters:
    X (np.ndarray): The dataset to search in. Shape (n, d), where n is the number of samples and d is the dimensionality.
    Q (np.ndarray): The query points. Shape (n_queries, d), where n_queries is the number of queries.'
    k (int): The number of nearest neighbors to find
    Returns:
    indices (np.ndarray): The indices of the nearest neighbors. Shape (n_queries, k).
    distances (np.ndarray): The distances to the nearest neighbors. Shape (n_queries, k).
    """
    index = faiss.IndexFlatL2(X.shape[1])
    index.add(X)

    distances, indices = index.search(Q, k)
    return indices, distances