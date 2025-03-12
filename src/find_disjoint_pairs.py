from functools import lru_cache
from math import comb
import numpy as np
import time
from tqdm import trange

# @lru_cache(maxsize=None)
def get_combs(D, s):
    return comb(D, s)

@lru_cache(maxsize=None)
def unrank_combination(D, s, idx):
    output = []
    x = 0
    for j in range(s):
        comb_value = get_combs(D - x - 1, s - j - 1)
        while comb_value <= idx:
            idx -= comb_value
            x += 1
            comb_value = get_combs(D - x - 1, s - j - 1)
        output.append(x)
        x += 1
    return output

def unrank_disjoint_pair(D, s, i):
    totalJ = get_combs(D - s, s)
    
    q = i // totalJ  # Index for I
    r = i % totalJ   # Index for J

    I = unrank_combination(D, s, q)
    remaining = [x for x in range(D) if x not in I]
    
    J = [remaining[j] for j in unrank_combination(D - s, s, r)]
    
    return I, J

if __name__ == "__main__":
    D, s = 20, 3
    
    comb_index = np.full((D+1, s+1), -1, dtype=object)

    start_time = time.time()
    
    total_pairs = get_combs(D, s) * get_combs(D - s, s)

    for i in trange(total_pairs):
        I, J = unrank_disjoint_pair(D, s, i)
        # print("I:", I, "J:", J) # I: [0, 1] J: [4, 5]

    print("Time:", time.time() - start_time)

    