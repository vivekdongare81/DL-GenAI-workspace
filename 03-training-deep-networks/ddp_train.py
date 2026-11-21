"""Data-parallel training with DistributedDataParallel (DDP).

Each process holds a full copy of the model and trains on its own slice of every epoch's data;
after each backward pass DDP averages the gradients across processes, so every copy stays identical.
Launch it with torchrun, which starts the processes and tells each one its rank:

    torchrun --nproc_per_node 2 ddp_train.py            # 2 processes on the CPU (backend "gloo")

On a machine with NVIDIA GPUs the same script runs one process per GPU with the "nccl" backend.
"""

import os

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset


def main() -> None:
    use_cuda = torch.cuda.is_available()
    dist.init_process_group(backend="nccl" if use_cuda else "gloo")    # reads RANK and WORLD_SIZE set by torchrun
    rank, world_size = dist.get_rank(), dist.get_world_size()
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}") if use_cuda else torch.device("cpu")

    torch.manual_seed(0)                                   # same data and same starting weights in every process
    X = torch.randn(4096, 20)
    y = (X[:, :5].sum(dim=1) > 0).long()
    dataset = TensorDataset(X, y)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=0)
    loader = DataLoader(dataset, batch_size=64, sampler=sampler)

    model = DDP(nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 2)).to(device))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(3):
        sampler.set_epoch(epoch)                           # a different shuffle every epoch, the same in all processes
        seen = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(xb), yb)
            loss.backward()                                # DDP averages the gradients across processes here
            optimizer.step()
            seen += len(yb)
        loss_avg = loss.detach().clone()
        dist.all_reduce(loss_avg)                          # sum the last-batch loss over processes, for logging
        if rank == 0:
            print(f"epoch {epoch}: each process trained on {seen} of {len(dataset)} examples, mean last-batch loss {loss_avg.item() / world_size:.4f}")

    checksum = torch.tensor([sum(p.sum().item() for p in model.parameters())])
    gathered = [torch.zeros(1) for _ in range(world_size)]
    dist.all_gather(gathered, checksum)
    if rank == 0:
        print("sum of all weights in each process:", [round(t.item(), 6) for t in gathered], "-> identical copies")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
