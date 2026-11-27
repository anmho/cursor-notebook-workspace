#!/usr/bin/env python3
"""A resumable, notebook-independent tiny training workload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ml_lab.device import select_device
from ml_lab.models.mlp import MLP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/tiny-train.pt"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = select_device(args.device)
    torch.manual_seed(args.seed)
    model = MLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    start_step = 0
    if args.resume and args.checkpoint.exists():
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"]) + 1
    inputs = torch.randn(32, 4, device=device)
    target = inputs.sum(dim=1, keepdim=True)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for step in range(start_step, start_step + args.steps):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(inputs), target)
        loss.backward()
        optimizer.step()
        if step == start_step or (step + 1) % 10 == 0:
            print(
                json.dumps(
                    {"step": step, "loss": float(loss.detach().cpu()), "device": str(device)}
                )
            )
    torch.save(
        {
            "step": start_step + args.steps - 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        args.checkpoint,
    )
    print(json.dumps({"checkpoint": str(args.checkpoint), "device": str(device)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
