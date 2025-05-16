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
    X_proj = np.array([
        [ 5.5,  5.5,  5.5,  5.5,  5.5,  5.5,  5.5,  5.5,  5.5,  5.5],
        [-4.5, -3.5, -2.5, -1.5, -0.5,  0.5,  1.5,  2.5,  3.5,  4.5],
        [ 1.0,  3.0,  5.0,  7.0,  9.0, 11.0, 13.0, 15.0, 17.0, 19.0],
    ])
    print(X_proj) 

    b = 2  # number of top values to keep

    index = s_ceos_indexing_s1(X_proj, b)
    
    flat = index.flatten()
    print(flat)
    print(len(flat))