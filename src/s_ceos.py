import numpy as np


def s_ceos_indexing_s1(X_proj, b: int):
    """
    Parameters:
        X: shape (d, n)         # input data
        R: shape (D, d)         # projection matrix
        b: int                  # number of top values to keep

    Returns:
        L_dict: dict of ((i, j): list of top-b indices)
    """
    D, n = X_proj.shape
    out = np.full((D, D, b), -1, dtype=int)

    for i in range(D):
        for j in range(D):
            if i == j:
                continue
            diff = X_proj[i] - X_proj[j]  # shape (n,)
            if b >= n:
                top_b = np.argsort(-diff)
            else:
                top_b = np.argpartition(-diff, b)[:b]
                top_b = top_b[np.argsort(-diff[top_b])]
            out[i, j] = top_b

    return out


if __name__ == "__main__":
    X_proj = np.array(
        [
            [0.534, 0.556, 0.545, 0.506, 0.455, 0.403, 0.357, 0.316, 0.283, 0.255],
            [-0.437, -0.354, -0.248, -0.138, -0.041, 0.037, 0.097, 0.144, 0.18, 0.208],
            [0.097, 0.303, 0.495, 0.644, 0.745, 0.807, 0.843, 0.863, 0.874, 0.88],
        ]
    )

    print(X_proj)

    b = 2  # number of top values to keep

    index = s_ceos_indexing_s1(X_proj, b)

    flat = index.flatten()
    print(flat)
    print(len(flat))
