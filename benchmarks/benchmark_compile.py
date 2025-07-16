import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from pathlib import Path
import argparse

# Ensure neuroexapt in path
import sys, os
sys.path.append(str(Path(__file__).resolve().parents[1]))

from neuroexapt.core.model import Network


def get_dataloader(batch_size: int, num_workers: int = 2):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    # Use small subset for quick benchmarking
    subset_indices = list(range(0, 1024))
    subset = Subset(dataset, subset_indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return loader


def build_model(use_checkpoint: bool, use_compile: bool):
    return Network(C=16, num_classes=10, layers=8, use_checkpoint=use_checkpoint, use_compile=use_compile).cuda()


def benchmark(model: nn.Module, loader: DataLoader, criterion, optimizer, warmup: int = 5, iters: int = 20):
    model.train()
    total_time = 0.0
    itr = 0
    for inputs, targets in loader:
        inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        start = time.perf_counter()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if itr >= warmup:
            total_time += elapsed
        itr += 1
        if itr >= warmup + iters:
            break
    avg = total_time / iters
    return avg


def main():
    parser = argparse.ArgumentParser(description="Benchmark compile+checkpoint vs baseline")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    loader = get_dataloader(args.batch_size)
    criterion = nn.CrossEntropyLoss()

    # Baseline model
    baseline = build_model(False, False)
    opt_base = optim.SGD(baseline.parameters(), lr=0.01, momentum=0.9)
    base_time = benchmark(baseline, loader, criterion, opt_base, iters=args.iters)
    print(f"Baseline avg step time: {base_time*1000:.2f} ms")

    # Optimized model (checkpoint + compile)
    optimized = build_model(True, True)
    opt_opt = optim.SGD(optimized.parameters(), lr=0.01, momentum=0.9)
    opt_time = benchmark(optimized, loader, criterion, opt_opt, iters=args.iters)
    print(f"Optimized avg step time: {opt_time*1000:.2f} ms")

    speedup = base_time / opt_time if opt_time > 0 else float('inf')
    print(f"Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    main() 