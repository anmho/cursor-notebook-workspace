"""Optional CuPy RawKernel example; requires Linux/NVIDIA and `uv sync --extra gpu`."""

import cupy as cp

kernel = cp.RawKernel(
    r"""
    extern "C" __global__ void square(const float* x, float* y, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) y[i] = x[i] * x[i];
    }
    """,
    "square",
)

x = cp.arange(1024, dtype=cp.float32)
y = cp.empty_like(x)
kernel((4,), (256,), (x, y, x.size))
print({"device": cp.cuda.runtime.getDevice(), "first": float(y[0]), "last": float(y[-1])})
