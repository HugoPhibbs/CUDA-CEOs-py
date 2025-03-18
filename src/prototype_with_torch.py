import time
from typing import List, Tuple
import torch
import tqdm
import itertools
import math


def disjoint_pairs(D: int, s: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    elements = range(D)
    all_pairs = []

    for I in tqdm(
        itertools.combinations(elements, s), "Creating Pairs", total=math.comb(D, s)
    ):
        remaining = set(elements) - set(I)
        for J in itertools.combinations(remaining, s):
            all_pairs.append(
                (torch.tensor(I, device="cuda"), torch.tensor(J, device="cuda"))
            )

    return all_pairs


def disjoint_pairs_to_gpu(
    all_pairs: List[Tuple[torch.Tensor, torch.Tensor]], s: int
) -> torch.Tensor:
    all_pairs_gpu = torch.tensor(len(all_pairs), 2 * s, device="cuda")

    for i, pair in enumerate(all_pairs):
        all_pairs_gpu[i, :] = torch.cat(pair).to("cuda")

    return all_pairs_gpu


def sum_projections_by_combs(X_prime: torch.Tensor, all_combs: torch.Tensor, d: int):
    sums = torch.full(size=(all_combs.shape[0], d), fill_value=-1, device="cuda")
    for i in range(all_combs.shape[0]):
        sums[i, :] = torch.sum(X_prime[all_combs[i, :], :], dim=0)
    return sums


def rank_combination_cpu(D: int, s: int, combination: list) -> int:
    """Compute the lexicographic rank of a sorted zero-based combination list."""
    rank = 0
    prev = -1  # Track the last selected number

    for i in range(s):
        x = combination[i]
        for j in range(prev + 1, x):  # Count skipped elements
            rank += math.comb(D - j - 1, s - i - 1)  # Compute skipped combinations
        prev = x  # Update last used number

    return rank


def create_index(
    combs_sums: torch.Tensor, all_pairs_gpu: torch.Tensor, D: int, s: int, b: int
) -> torch.Tensor:
    D_choose_s = math.comb(D, s)
    D_minus_s_choose_s = math.comb(D - s, s)
    index = torch.full(size=(D_choose_s, D_choose_s, b), fill_value=-1, device="cuda")

    for i in range(all_pairs_gpu.shape[0]):
        J = all_pairs_gpu[i, s:]

        I_idx = i // D_minus_s_choose_s

        rank_J = rank_combination_cpu(D, s, J.cpu().tolist())

        index[I_idx, rank_J, :] = torch.topk(
            combs_sums[I_idx, :] - combs_sums[rank_J], b
        ).indices

    return index


def indexing(X: torch.Tensor, D: int, s: int, b: int, k: int):
    d = X.shape[1]
    R = torch.normal(0, 1, size=(D, d))

    X_prime = torch.mm(R, X)

    all_pairs = disjoint_pairs(D, s)

    all_pairs_gpu = disjoint_pairs_to_gpu(all_pairs, s)

    combs_sums = sum_projections_by_combs(X_prime, all_pairs_gpu, d)

    index = create_index(combs_sums, all_pairs_gpu, D, s, b)

    return index


def querying(index: torch.Tensor, R: torch.Tensor, q: torch.Tesnor, k: int, D: int, s: int):
    q_prime = torch.mm(R, q)

    I = torch.topk(q_prime, s).indices
    J = torch.topk(q_prime, s, largest=False).indices

    rank_I = rank_combination_cpu(D, s, I.cpu().tolist())
    rank_J = rank_combination_cpu(D, s, J.cpu().tolist())

    index_vecs = index[rank_I, rank_J, :]

    inner_products = torch.mm(index_vecs, q_prime)

    topk_indices = torch.topk(inner_products, k).indices

    return topk_indices


if __name__ == "__main__":
    # Check GPU is available
    if torch.cuda.is_available():
        print("CUDA is available")
    else:
        print("CUDA is not available")
        exit()

    
