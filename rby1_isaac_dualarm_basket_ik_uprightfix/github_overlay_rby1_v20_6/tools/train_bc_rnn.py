#!/usr/bin/env python3
"""Small behavior-cloning GRU trainer for RB-Y1 v7 demonstration datasets.

The trained checkpoint is intentionally not applied directly to the robot by the
v7 controller. Validate a learned policy offline before adding it as a bounded
residual around the deterministic 6D IK controller.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class BCRNN(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, action_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.gru(x)
        return self.head(hidden)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    args = parser.parse_args()

    data = np.load(args.dataset.expanduser())
    observations = torch.from_numpy(data["observations"].astype(np.float32))
    actions = torch.from_numpy(data["actions"].astype(np.float32))
    count = observations.shape[0]
    if count < 10:
        raise RuntimeError("At least 10 sequences are required; 50+ successful sequences are recommended.")

    generator = torch.Generator().manual_seed(7)
    permutation = torch.randperm(count, generator=generator)
    split = max(1, int(0.9 * count))
    train_indices, valid_indices = permutation[:split], permutation[split:]
    if valid_indices.numel() == 0:
        valid_indices = train_indices[-1:]

    train_loader = DataLoader(
        TensorDataset(observations[train_indices], actions[train_indices]),
        batch_size=args.batch_size,
        shuffle=True,
    )
    valid_x, valid_y = observations[valid_indices], actions[valid_indices]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BCRNN(observations.shape[-1], actions.shape[-1], args.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    best = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        samples = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss) * x.shape[0]
            samples += x.shape[0]
        model.eval()
        with torch.no_grad():
            valid_loss = float(loss_fn(model(valid_x.to(device)), valid_y.to(device)))
        if valid_loss < best:
            best = valid_loss
            args.output.expanduser().parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "input_dim": observations.shape[-1],
                    "action_dim": actions.shape[-1],
                    "hidden_dim": args.hidden,
                    "validation_mse": best,
                },
                args.output.expanduser(),
            )
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"epoch={epoch:03d} train_mse={total/max(samples,1):.6f} valid_mse={valid_loss:.6f}")
    print(f"[OK] best validation MSE={best:.6f}; checkpoint={args.output.expanduser()}")


if __name__ == "__main__":
    main()
