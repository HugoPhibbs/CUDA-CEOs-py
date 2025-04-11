import numpy as np
import os
import re
from tqdm import tqdm

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
            data[i] = np.fromstring(line, sep=" ", dtype=np.float32)  # Parse efficiently

    os.makedirs(os.path.dirname(output_file), exist_ok=True)  # Ensure directory exists
    data.tofile(output_file)
    print(f"Wrote {n}x{d} data to {output_file}")

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

    bin_filenames = [
        "/workspace/CUDA-CEOs/datasets/bin/imagenet/imagenet_queries.bin",
        "/workspace/CUDA-CEOs/datasets/bin/imagenet/imagenet_dataset.bin",
        "/workspace/CUDA-CEOs/datasets/bin/msong/msong_queries.bin",
        "/workspace/CUDA-CEOs/datasets/bin/msong/msong_dataset.bin",
        "/workspace/CUDA-CEOs/datasets/bin/nuswide/nuswide_queries.bin",
        "/workspace/CUDA-CEOs/datasets/bin/nuswide/nuswide_dataset.bin",
        "/workspace/CUDA-CEOs/datasets/bin/yahoo/yahoo_queries.bin",
        "/workspace/CUDA-CEOs/datasets/bin/yahoo/yahoo_dataset.bin"
    ]

    for txt_filename, bin_filename in tqdm(zip(txt_filenames, bin_filenames), "Writing files"):
        dataset_txt_to_binary(txt_filename, bin_filename)

    # array = np.fromfile("/workspace/CUDA-CEOs/datasets/imagenet/imagenet_dataset.bin", dtype=np.float32)
    # array = array.reshape(-1, 150)
    # print(array.shape)
