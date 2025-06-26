import os
import pandas as pd 
import requests
import h5py

glove100_url = r"http://ann-benchmarks.com/glove-100-angular.hdf5"
local_path = "/workspace/CUDA-CEOs/datasets/hdf5/glove-100-angular.hdf5"

if not os.path.exists(os.path.dirname(local_path)):
    with open(local_path, "wb") as f:
        response = requests.get(glove100_url)
        f.write(response.content)

with h5py.File(local_path, 'r') as f:
    # List all groups
    print("Keys: %s" % f.keys())

    train_data = f['train'][:]
    test_data = f['test'][:]


dataset_dir = r"/workspace/CUDA-CEOs/datasets/bin/glove-100"

if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

train_data.tofile(f"{dataset_dir}/glove-100_dataset.bin")
test_data.tofile(f"{dataset_dir}/glove-100_queries.bin")