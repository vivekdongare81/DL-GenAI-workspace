"""Fast tests for the training code. Run from this folder: python -m pytest -q"""

import numpy as np
import pytest
import torch
from pydantic import ValidationError
from torch import nn

from data import FashionMNISTDataset
from model import MLP
from train import Config, main


def test_model_output_shape():
    assert MLP([32])(torch.rand(4, 28, 28)).shape == (4, 10)


def test_dataset_returns_float_image_and_int64_label():
    dataset = FashionMNISTDataset(np.full((3, 28, 28), 255, np.uint8), np.array([1, 2, 3], np.uint8))
    image, label = dataset[0]
    assert len(dataset) == 3
    assert image.dtype == torch.float32 and image.max() == 1.0
    assert label.dtype == torch.int64


def test_config_rejects_typos_and_bad_values():
    with pytest.raises(ValidationError):
        Config(data_dir="data", learning_rate=0.1)  # typo: the field is lr
    with pytest.raises(ValidationError):
        Config(data_dir="data", lr=-1)


def test_model_can_overfit_one_batch():
    """If the model can't memorise 32 examples, something in the training code is broken."""
    torch.manual_seed(0)
    x, y = torch.rand(32, 28, 28), torch.randint(0, 10, (32,))
    model = MLP([64])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(100):
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
    assert loss.item() < 0.05


def test_checkpoint_round_trip(tmp_path):
    model = MLP([32], mean=0.3, std=0.4).eval()
    torch.save(model.state_dict(), tmp_path / "model.pt")
    restored = MLP([32]).eval()
    restored.load_state_dict(torch.load(tmp_path / "model.pt", weights_only=True))
    x = torch.rand(2, 28, 28)
    assert torch.equal(model(x), restored(x))


def test_one_epoch_end_to_end(tmp_path):
    """The whole script on the real data for one epoch: it must finish and write its outputs."""
    config = tmp_path / "config.yaml"
    config.write_text(f"data_dir: ../../data/fashion-mnist\nout_dir: {tmp_path}\nepochs: 1\nnum_workers: 0\ndevice: cpu\n")
    main(["--config", str(config), "--run-name", "smoke"])
    assert {"best.pt", "last.pt", "metrics.json", "train.log", "config.yaml"} <= {p.name for p in (tmp_path / "smoke").iterdir()}
