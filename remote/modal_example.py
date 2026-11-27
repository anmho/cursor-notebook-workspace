"""Optional Modal example; run only after approving cloud execution and authenticating Modal.

    uv sync --extra remote
    modal token new
    modal run remote/modal_example.py

The volume is created/attached only when this file is executed through Modal.
"""

from __future__ import annotations

import os

import modal

app = modal.App("ml-lab")
image = modal.Image.debian_slim(python_version="3.12").pip_install("torch")


@app.function(
    image=image,
    gpu=os.getenv("MODAL_GPU", "T4"),
    timeout=300,
)
def cuda_smoke_test() -> dict[str, object]:
    import torch

    device = torch.device("cuda")
    left = torch.randn(1024, 1024, device=device)
    right = torch.randn(1024, 1024, device=device)
    result = left @ right
    return {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0),
        "result_shape": list(result.shape),
    }


@app.local_entrypoint()
def main() -> None:
    print(cuda_smoke_test.remote())
