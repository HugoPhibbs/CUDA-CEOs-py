import pandas as pd
import numpy as np
import os

from write_load_datasets import DATASET_DIR

sift_data_df = pd.read_parquet("hf://datasets/open-vdb/sift-128-euclidean/train/train-00001-of-00001.parquet")
sift_query_df = pd.read_parquet("hf://datasets/open-vdb/sift-128-euclidean/test/test-00001-of-00001.parquet")

sift_data = np.stack(sift_data_df["emb"].to_numpy()).astype(np.float32)
sift_query = np.stack(sift_query_df["emb"].to_numpy()).astype(np.float32)

os.makedirs(os.path.join(DATASET_DIR, "bin", "sift-128"), exist_ok=True)

sift_data_file = os.path.join(DATASET_DIR, "bin", "sift-128", "sift-128_dataset.bin")

sift_query_file = os.path.join(DATASET_DIR, "bin", "sift-128", "sift-128_queries.bin")

sift_data.tofile(sift_data_file)
sift_query.tofile(sift_query_file)




