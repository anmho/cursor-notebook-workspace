"""Optional Triton example; requires Linux/NVIDIA and `uv sync --extra gpu`."""

import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, block_size: tl.constexpr):
    block = tl.program_id(0)
    offsets = block * block_size + tl.arange(0, block_size)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


if __name__ == "__main__":
    x = torch.ones(1024, device="cuda")
    y = torch.full_like(x, 2)
    out = torch.empty_like(x)
    add_kernel[(triton.cdiv(x.numel(), 256),)](x, y, out, x.numel(), block_size=256)
    print({"device": str(out.device), "first": float(out[0]), "last": float(out[-1])})
