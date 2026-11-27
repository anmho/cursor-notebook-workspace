"""Device selection and a small end-to-end PyTorch smoke test."""

from __future__ import annotations

import platform
from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class DeviceReport:
    requested: str
    selected: str
    torch_version: str
    python_architecture: str
    cuda_available: bool
    mps_available: bool
    cuda_device: str | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _mps_available() -> bool:
    return bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())


def select_device(requested: str = "auto") -> torch.device:
    """Select a device without silently falling back for an explicit request."""

    normalized = requested.lower().strip()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _mps_available():
            return torch.device("mps")
        return torch.device("cpu")
    if normalized in {"cuda", "gpu"}:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was explicitly requested, but torch.cuda.is_available() is false."
            )
        return torch.device("cuda")
    if normalized == "mps":
        if not _mps_available():
            raise RuntimeError("MPS was explicitly requested, but the MPS backend is unavailable.")
        return torch.device("mps")
    if normalized == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device {requested!r}; use auto, cuda, mps, or cpu.")


def device_report(requested: str = "auto") -> DeviceReport:
    device = select_device(requested)
    cuda_device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return DeviceReport(
        requested=requested,
        selected=str(device),
        torch_version=torch.__version__,
        python_architecture=platform.machine(),
        cuda_available=torch.cuda.is_available(),
        mps_available=_mps_available(),
        cuda_device=cuda_device,
    )


def run_smoke_test(requested: str = "auto", steps: int = 5) -> dict[str, object]:
    """Allocate, multiply, backpropagate, and train a tiny model."""

    device = select_device(requested)
    torch.manual_seed(7)
    inputs = torch.randn(16, 4, device=device)
    weights = torch.randn(4, 4, device=device)
    matmul = inputs @ weights
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 1)).to(
        device
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    target = torch.randn(16, 1, device=device)
    losses: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(inputs), target)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return {
        "device": str(device),
        "matmul_shape": list(matmul.shape),
        "steps": steps,
        "losses": losses,
        "final_loss": losses[-1],
    }
