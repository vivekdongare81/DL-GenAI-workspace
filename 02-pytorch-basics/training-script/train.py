"""Train the FashionMNIST MLP from the command line. Run from this folder:

python train.py --config config.yaml
python train.py --config config.yaml --epochs 3 --lr 0.003 --run-name quick-test
python train.py --config config.yaml --resume checkpoints/quick-test/last.pt --epochs 6
"""

import argparse
import json
import logging
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field
from torch import nn

from data import make_loaders
from model import MLP

logger = logging.getLogger("train")


class Config(BaseModel):
    """Every setting of a run. Unknown keys and invalid values fail before training starts."""

    model_config = ConfigDict(extra="forbid")

    data_dir: Path
    out_dir: Path = Path("checkpoints")
    seed: int = 0
    device: str = "auto"
    epochs: int = Field(15, ge=1)
    batch_size: int = Field(128, ge=1)
    lr: float = Field(1e-3, gt=0)
    hidden: list[int] = [256, 128]
    dropout: float = Field(0.2, ge=0, lt=1)
    val_size: int = Field(6000, gt=0)
    patience: int = Field(3, ge=1)
    num_workers: int = Field(2, ge=0)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an MLP on FashionMNIST.")
    parser.add_argument("--config", type=Path, required=True, help="YAML file with the run settings")
    parser.add_argument("--epochs", type=int, help="override the number of epochs")
    parser.add_argument("--lr", type=float, help="override the learning rate")
    parser.add_argument("--batch-size", type=int, help="override the batch size")
    parser.add_argument("--run-name", help="folder name for this run (default: date and time)")
    parser.add_argument("--resume", type=Path, help="continue from a last.pt checkpoint")
    return parser.parse_args(argv)


def load_config(args: argparse.Namespace) -> Config:
    """YAML values first, then command-line overrides; Pydantic validates the result."""
    values = yaml.safe_load(args.config.read_text())
    overrides = {"epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size}
    values.update({key: value for key, value in overrides.items() if value is not None})
    return Config(**values)


def setup_logging(log_file: Path) -> None:
    """Log to the console and to a file in the run folder."""
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_device_rng(device: torch.device) -> torch.Tensor | None:
    """GPU operations such as dropout use the GPU's own random generator, separate from the CPU one."""
    if device.type == "cuda":
        return torch.cuda.get_rng_state()
    if device.type == "mps":
        return torch.mps.get_rng_state()
    return None


def set_device_rng(device: torch.device, state: torch.Tensor | None) -> None:
    if device.type == "cuda":
        torch.cuda.set_rng_state(state.cpu())
    elif device.type == "mps":
        torch.mps.set_rng_state(state.cpu())


def git_commit() -> str:
    """The code version this run used, so a result can be traced back to its code."""
    try:
        result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def train_one_epoch(model, loader, loss_fn, optimizer, device) -> float:
    model.train()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(y)
        n += len(y)
    return total / n


@torch.inference_mode()
def evaluate(model, loader, loss_fn, device) -> tuple[float, float]:
    model.eval()
    total, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total += loss_fn(logits, y).item() * len(y)
        correct += (logits.argmax(dim=1) == y).sum().item()
        n += len(y)
    return total / n, correct / n


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = load_config(args)
    run_dir = args.resume.parent if args.resume else cfg.out_dir / (args.run_name or datetime.now().strftime("%Y%m%d-%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir / "train.log")
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False))

    set_seed(cfg.seed)
    device = get_device(cfg.device)
    logger.info("run %s | torch %s | device %s | git %s", run_dir.name, torch.__version__, device, git_commit())

    shuffle_generator = torch.Generator()
    train_loader, val_loader, test_loader, mean, std = make_loaders(
        cfg.data_dir, cfg.batch_size, cfg.val_size, cfg.num_workers, shuffle_generator
    )
    model = MLP(cfg.hidden, cfg.dropout, mean, std).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.CrossEntropyLoss()
    start_epoch, best_acc, stale_epochs, history = 1, 0.0, 0, []

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        torch.set_rng_state(ckpt["torch_rng"].cpu())  # map_location moved it; RNG states live on the CPU
        set_device_rng(device, ckpt["device_rng"])
        start_epoch, best_acc, stale_epochs, history = ckpt["epoch"] + 1, ckpt["best_acc"], ckpt["stale_epochs"], ckpt["history"]
        logger.info("resumed from %s after epoch %d (best val_acc %.4f)", args.resume, ckpt["epoch"], best_acc)

    for epoch in range(start_epoch, cfg.epochs + 1):
        start = time.perf_counter()
        shuffle_generator.manual_seed(cfg.seed + epoch)  # the batch order depends only on (seed, epoch)
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc})
        logger.info(
            "epoch %d/%d train_loss %.4f val_loss %.4f val_acc %.4f (%.1fs)",
            epoch,
            cfg.epochs,
            train_loss,
            val_loss,
            val_acc,
            time.perf_counter() - start,
        )

        if val_acc > best_acc:  # model checkpoint: the best weights, for serving
            best_acc, stale_epochs = val_acc, 0
            torch.save(
                {"model": model.state_dict(), "config": cfg.model_dump(mode="json"), "epoch": epoch, "val_acc": val_acc},
                run_dir / "best.pt",
            )
        else:
            stale_epochs += 1
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,  # training checkpoint
                "best_acc": best_acc,
                "stale_epochs": stale_epochs,
                "history": history,
                "torch_rng": torch.get_rng_state(),
                "device_rng": get_device_rng(device),
            },
            run_dir / "last.pt",
        )
        if stale_epochs >= cfg.patience:
            logger.info("early stopping: no better val_acc for %d epochs", cfg.patience)
            break

    best = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(best["model"])
    _, test_acc = evaluate(model, test_loader, loss_fn, device)
    logger.info("best epoch %d: val_acc %.4f, test_acc %.4f", best["epoch"], best["val_acc"], test_acc)
    metrics = {"best_epoch": best["epoch"], "val_acc": best["val_acc"], "test_acc": test_acc, "history": history}
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
