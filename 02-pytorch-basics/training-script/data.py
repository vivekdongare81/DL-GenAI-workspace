"""FashionMNIST: download with checksum checks, a Dataset, and the DataLoaders."""

import gzip
import hashlib
import urllib.request
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, random_split

BASE_URL = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
FILES = {
    "train-images-idx3-ubyte.gz": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
    "train-labels-idx1-ubyte.gz": "25c81989df183df01b3e8a0aad5dffbe",
    "t10k-images-idx3-ubyte.gz": "bef4ecab320f06d8554ea6380940ec79",
    "t10k-labels-idx1-ubyte.gz": "bb300cfdad3c16e7a12a480ee83cd310",
}
CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def download(data_dir: Path) -> None:
    """Download each file once; delete it and stop if its checksum is wrong."""
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, expected_md5 in FILES.items():
        path = data_dir / name
        if not path.exists():
            urllib.request.urlretrieve(BASE_URL + name, path)
        if hashlib.md5(path.read_bytes()).hexdigest() != expected_md5:
            path.unlink()
            raise ValueError(f"{name}: checksum mismatch, file deleted")


def read_idx(path: Path) -> np.ndarray:
    with gzip.open(path) as f:
        data = f.read()
    if "images" in path.name:
        return np.frombuffer(data, np.uint8, offset=16).reshape(-1, 28, 28).copy()
    return np.frombuffer(data, np.uint8, offset=8).copy()


class FashionMNISTDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray):
        self.images = torch.from_numpy(images)
        self.labels = torch.from_numpy(labels).long()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images[i].float() / 255, self.labels[i]


def make_loaders(data_dir: Path, batch_size: int, val_size: int, num_workers: int, shuffle_generator: torch.Generator):
    """Train, validation and test DataLoaders, plus the pixel mean and std of the training split.
    The training order comes from shuffle_generator; reseed it every epoch (seed + epoch) so a resumed run sees the same batches."""
    download(data_dir)
    full_train = FashionMNISTDataset(read_idx(data_dir / "train-images-idx3-ubyte.gz"), read_idx(data_dir / "train-labels-idx1-ubyte.gz"))
    test_set = FashionMNISTDataset(read_idx(data_dir / "t10k-images-idx3-ubyte.gz"), read_idx(data_dir / "t10k-labels-idx1-ubyte.gz"))
    split_rng = torch.Generator().manual_seed(0)  # the same split in every run
    train_set, val_set = random_split(full_train, [len(full_train) - val_size, val_size], generator=split_rng)

    pixels = full_train.images[train_set.indices].float() / 255
    workers = {"num_workers": num_workers, "persistent_workers": num_workers > 0}
    sampler = RandomSampler(train_set, generator=shuffle_generator)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, sampler=sampler, drop_last=True, generator=torch.Generator().manual_seed(0), **workers
    )  # seeds the worker processes
    val_loader = DataLoader(val_set, batch_size=512, **workers)
    test_loader = DataLoader(test_set, batch_size=512, **workers)
    return train_loader, val_loader, test_loader, pixels.mean().item(), pixels.std().item()
