import numpy as np

def sCEOs_TA_indexing(X :  np.ndarray, R: np.ndarray):
    D = R.shape[0]
    n = X.shape[1]

    X_prime = R @ X

    index = np.empty(shape=(D, n))

    for j in range(D):
        index[j, :] = np.argsort(X_prime[j, :])

    return index

def sCEOs_TA_querying(q: np.ndarray, X: np.ndarray, R: np.ndarray, index: np.ndarray, s_0: int):
    q_prime = q @ R

    n = X.shape[1]

    max_dims = np.argpartition(q_prime, -s_0)[-s_0:][::-1]
    min_dims = np.argpartition(q_prime, s_0)[:s_0]

    for i in range(n):
        threshold = 0 

        for j in range(s_0):
            this_max_dim = max_dims[j]
            this_min_dim = min_dims[j]

            far_point_idx = index[this_max_dim, i]
            near_point_idx = index[this_min_dim, n - i - 1]

            threshold += 

    