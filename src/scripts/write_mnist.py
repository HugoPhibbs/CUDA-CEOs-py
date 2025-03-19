import numpy as np
import torchvision
import os

MNIST_DATA_DIR = "/workspace/CUDA-CEOs/datasets/bin/MNIST"
DATASET_CACHE_DIR = "/workspace/CUDA-CEOs/datasets/torch-cache"

FILE_MAP = {
    "float32": (f"{MNIST_DATA_DIR}/mnist_images_row_major.bin", f"{MNIST_DATA_DIR}/mnist_labels.bin"),
    "float16": (f"{MNIST_DATA_DIR}/mnist_images_row_major_f16.bin", f"{MNIST_DATA_DIR}/mnist_labels_f16.bin"),
}

def write_mnist_to_binary(shuffle=True, dtype="float32"):
    datasets = [
        torchvision.datasets.MNIST(DATASET_CACHE_DIR, train=split, download=True) 
        for split in (True, False)
    ]
    
    images = np.concatenate([ds.data.numpy().astype(dtype) for ds in datasets])
    labels = np.concatenate([ds.targets.numpy().astype(np.uint8) for ds in datasets])

    if shuffle:
        indices = np.random.permutation(images.shape[0])
        images, labels = images[indices], labels[indices]

    if not os.path.exists(MNIST_DATA_DIR):
        os.makedirs(MNIST_DATA_DIR)

    images.reshape(images.shape[0], -1).tofile(FILE_MAP[dtype][0])

    labels.tofile(FILE_MAP[dtype][1])

    print("Wrote MNIST data to binary files")

    return FILE_MAP[dtype], images, labels

if __name__ == "__main__":
    write_mnist_to_binary(dtype="float32")
