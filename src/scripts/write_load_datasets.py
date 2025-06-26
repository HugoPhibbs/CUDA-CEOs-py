import numpy as np
import os
import re
from tqdm import tqdm

DATASET_DIR = "/workspace/CUDA-CEOs/datasets"

DATASET_INFO = {
    "imagenet": {
        "n": 2340373,
        "d": 150,
        "n_queries": 200
    },
    "msong": {
        "n": 994185,
        "d": 420,
        "n_queries": 1000
    },
    "nuswide": {
        "n": 268643,
        "d": 500,
        "n_queries": 200
    },
    "yahoo": {
        "n": 624961,
        "d": 300,
        "n_queries": 1000
    }, 
    "sift-128": {
        "n": 1_000_000,  # Example value, adjust as needed
        "d": 128,
        "n_queries": 10000  # Example value, adjust as needed
    }, 
    "glove-100": {
        "n": 1_183_514,  # Example value, adjust as needed
        "d": 100,
        "n_queries": 10000  # Example value, adjust as needed
    },
}

def parse_n_d_from_filename(filename):
    """Extracts (n, d) from a filename of the form *_n_d.txt"""
    match = re.search(r'_(\d+)_(\d+)', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    raise ValueError(f"Could not parse (n, d) from filename: {filename}")

def dataset_txt_to_binary(input_file, output_file):
    n, d = parse_n_d_from_filename(input_file)  # Extract expected shape
    data = np.empty((n, d), dtype=np.float32)  # Preallocate array

    with open(input_file, "r") as f:
        for i, line in tqdm(enumerate(f), "Reading File", total=n):
            data[i] = np.fromstring(line, sep=" ", dtype=np.float32) 

    os.makedirs(os.path.dirname(output_file), exist_ok=True)  # Ensure directory exists
    data.tofile(output_file)
    print(f"Wrote {n}x{d} data to {output_file}")

def dataset_bin_paths(dataset_name:str):
    bin_path = os.path.join(DATASET_DIR, "bin", dataset_name)
    return [
        os.path.join(bin_path, f"{dataset_name}_dataset.bin"),
        os.path.join(bin_path, f"{dataset_name}_queries.bin"),
    ]   


def load_dataset(dataset_name: str):
    """
    Load dataset and queries from binary files.

    Parameters:
    dataset_name (str): Name of the dataset to load. Should be one of the keys in DATA_SET_INFO.
    Returns:
    tuple: A tuple containing the dataset and queries as numpy arrays, in that order.
    """
    dataset_path, query_path = dataset_bin_paths(dataset_name)
    dataset_info = DATASET_INFO[dataset_name]
    n, d, n_queries = dataset_info["n"], dataset_info["d"], dataset_info["n_queries"]

    dataset = np.fromfile(dataset_path, dtype=np.float32)
    dataset = dataset.reshape((n, d))

    queries = np.fromfile(query_path, dtype=np.float32)
    queries = queries.reshape((n_queries, d))

    return dataset, queries

if __name__ == "__main__":
    txt_filenames = [
        "/workspace/CUDA-CEOs/datasets/txt/imagenet/_Q_200_150.txt",
        "/workspace/CUDA-CEOs/datasets/txt/imagenet/_X_2340373_150.txt",
        "/workspace/CUDA-CEOs/datasets/txt/msong/_Q_1000_420.txt",
        "/workspace/CUDA-CEOs/datasets/txt/msong/_X_994185_420.txt",
        "/workspace/CUDA-CEOs/datasets/txt/nuswide/_Q_200_500.txt",
        "/workspace/CUDA-CEOs/datasets/txt/nuswide/_X_268643_500.txt",
        "/workspace/CUDA-CEOs/datasets/txt/yahoo/Yahoo_Q_1000_300.txt",
        "/workspace/CUDA-CEOs/datasets/txt/yahoo/Yahoo_X_624961_300.txt"
    ]

    datasets = [
        "imagenet",
        "msong",
        "nuswide",
        "yahoo"
    ]

    bin_filenames = []
    for dataset_name in datasets:
        bin_filenames += dataset_bin_paths(dataset_name)
    
    for txt_filename, bin_filename in tqdm(zip(txt_filenames, bin_filenames), "Writing files"):
        dataset_txt_to_binary(txt_filename, bin_filename)

    # array = np.fromfile("/workspace/CUDA-CEOs/datasets/imagenet/imagenet_dataset.bin", dtype=np.float32)
    # array = array.reshape(-1, 150)
    # print(array.shape)
