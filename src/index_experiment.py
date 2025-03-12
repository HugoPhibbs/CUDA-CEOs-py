import torch
from tqdm import tqdm
import itertools
import math

def random_dataset(D, d):
    return torch.normal(0, 1, size=(D, d))

def disjoint_pairs(D, s):
    elements = range(D) 
    all_pairs = []

    for I in tqdm(itertools.combinations(elements, s), "Creating Pairs", total=math.comb(D, s)):
        remaining = set(elements) - set(I)
        for J in itertools.combinations(remaining, s):
            all_pairs.append((torch.tensor(I), torch.tensor(J)))


def indexing(X, R, D, s0, b):
    X_prime = torch.mm(R, X)

    all_pairs = disjoint_pairs(D, s0)

    index = {}

    for (I, J) in tqdm(all_pairs, "Performing Indexing"):
        summed = torch.sum(X_prime[I, :], dim=0) - torch.sum(X_prime[J, :], dim=0)
        _, top_b_indices = torch.topk(summed, b)

        index[(tuple(torch.sort(I)), tuple(torch.sort(J)))] = top_b_indices

    return index

