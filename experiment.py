import itertools
import math

import torchvision
import torch
import random
import cupy as cp
import pickle

from tqdm import tqdm

if not torch.cuda.is_available():
    print("CUDA is not available. Exiting...")
    exit(1)
else:
    print("🎉 CUDA is available.")

DISJOINT_PAIRS_DIR=r"C:\Users\hugop\Documents\Uni\CUDA-CEOs\data\disjoint_pairs"

def load_mnist_data():
    mnist = torchvision.datasets.mnist.MNIST(root='./data', train=True, download=True)

    X = mnist.data

    d = 28 * 28

    X = X.reshape(-1, d).t()
    X = X[:, :1000]
    X = X / 255.0

    return X, d


def downproject_dataset(X, svd_dim=3):
    U, S, V = torch.svd(X.t())  # Assume X is a d x n matrix

    X_projected = U[:, :svd_dim] @ torch.diag(S[:svd_dim])

    X_projected = X_projected.t()

    max_col_norm = torch.max(torch.norm(X_projected, dim=0))

    X_projected = X_projected / max_col_norm

    return X_projected, svd_dim


def random_dataset(D, d):
    return torch.normal(0, 1, size=(D, d))


def all_disjoint_lists(D, s, save_dir=DISJOINT_PAIRS_DIR):
    elements = range(D)  # Implicit list [0, 1, ..., D-1]
    all_pairs = []

    save_path = None

    if save_dir:
        save_path = f"{save_dir}/disjoint_pairs_{D}_{s}.pkl"
        try:
            with open(save_path, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            print("File not found. Creating pairs...")
            pass

    for I in tqdm(itertools.combinations(elements, s), "Creating Pairs", total=math.comb(D, s)):
        remaining = set(elements) - set(I)
        for J in itertools.combinations(remaining, s):
            all_pairs.append((torch.tensor(I), torch.tensor(J)))

    if save_path:
        all_pairs_cpu = []

        for I, J in tqdm(all_pairs, "Converting pairs to CPU"):
            all_pairs_cpu.append((I.cpu(), J.cpu()))

        with open(save_path, "wb") as f:

            pickle.dump(all_pairs_cpu, f)

            print(f"Saved pairs to {save_path}")

    return all_pairs


def indexing(X, R, D, s0, b):
    X_prime = torch.mm(R, X)

    all_pairs = all_disjoint_lists(D, s0)

    index = {}

    for (I, J) in tqdm(all_pairs, "Performing Indexing"):
        summed = torch.sum(X_prime[I, :], dim=0) - torch.sum(X_prime[J, :], dim=0)
        _, top_b_indices = torch.topk(summed, b)

        index[(tuple(torch.sort(I)), tuple(torch.sort(J)))] = top_b_indices

    return index


def querying(q, X, R, index, s0, k):
    q_prime = torch.mm(R, q)
    _, I = torch.topk(q_prime, s0) # Fix this line here
    _, J = torch.topk(q_prime, s0, largest=False)

    top_b_indices = index[(tuple(torch.sort(I)), tuple(torch.sort(J)))]

    top_b_projections = q @ X[:, top_b_indices]

    top_k_projections, top_k_indices = torch.topk(top_b_projections, k)

    return top_k_projections, top_k_indices


if __name__ == "__main__":
    X_raw, _ = load_mnist_data()
    d = 10
    X, _ = downproject_dataset(X_raw, svd_dim=d)

    D = 14
    s0 = 4

    b = 5

    k = 3

    R = random_dataset(D, d)

    index = indexing(X, R, D, s0, b)

    q = torch.normal(0, 1, size=(d, 1))

    top_k_projections, top_k_indices = querying(q, X, R, index, s0, k)

    print(top_k_projections)
    print(top_k_indices)



