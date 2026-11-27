# ML lab

This is a small, reproducible workspace for ML engineering education, RL, systems, kernels, and remote GPU work. Cursor + standard Python + Jupyter is the notebook workflow; regular Python modules and CUDA/C++ source remain the implementation surface. The local baseline targets Python 3.12 on Apple Silicon and uses PyTorch's MPS backend when available. On Linux, uv selects the official CUDA 13.0 PyTorch index configured in `pyproject.toml`.

## First run

```bash
cd ~/Developer/ml-lab
uv python install 3.12
uv sync
uv run python scripts/doctor.py
```

Standard workflow:

```bash
uv sync
uv run python scripts/train.py --device auto --steps 100
uv run pytest
uv run ruff check .
uv run ruff format .
```

`auto` chooses CUDA, then MPS, then CPU. `--device cuda` and `--device mps` are explicit and fail loudly when unavailable.

## Notebook workflows

- Jupyter notebook: open `notebooks/00_environment_check.ipynb`, select `ML Lab (.venv)`, and run cells with `Shift+Enter`. Use `Run All`, `Restart Kernel`, and `Clear All Outputs` from the notebook command palette.
- Python cells: open `notebooks/01_tensor_playground.py` or `notebooks/02_training_visualization.py`; use the `# %%` controls or `Shift+Enter`. Both files also run from the terminal.
- Cursor: use `Python: Select Interpreter` and choose `.venv/bin/python`; use `Jupyter: Select Notebook Kernel` for notebooks and `Remote-SSH: Connect to Host...` for a remote workspace.

## GPU programming playground

The `kernels/` directory contains intentionally small examples: a standalone CUDA vector-add kernel, a CuPy `RawKernel`, a Triton kernel, and a PyTorch reference. They are remote-Linux examples and are not installed or executed by the Mac bootstrap.

```bash
uv sync --extra gpu       # Linux/NVIDIA only
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
uv run python kernels/cupy_square.py
uv run python kernels/triton_add.py
```

On a remote box, compare the custom kernels to the PyTorch reference before profiling. Use `cuda-gdb`, Compute Sanitizer, Nsight Systems, and Nsight Compute only after verifying the provider permits debugger access and performance counters.

## Remote SSH GPU workflow

For a host configured in `~/.ssh/config`:

```bash
ssh GPU_HOST
git clone <your-repository-url>
curl -LsSf https://astral.sh/uv/install.sh | sh
cd ml-lab
uv python install 3.12
uv sync
uv run python scripts/doctor.py
nvidia-smi
```

Open the remote directory in Cursor with `Remote-SSH: Connect to Host...`. Select the remote `.venv/bin/python` interpreter and the remote Jupyter kernel. Keep long jobs alive with:

```bash
tmux new -s ml
uv run python scripts/train.py --device cuda --steps 10000
# detach: Ctrl-b then d; reattach: tmux attach -t ml
```

The Linux CUDA wheel is selected by the platform marker in `pyproject.toml`; the NVIDIA driver, CUDA runtime/toolkit, and PyTorch wheel are separate concerns. Check the remote driver and toolkit with `nvidia-smi` and `nvcc --version` before adding Triton, Nsight, NCCL, or FlashAttention.

## Remote Jupyter with a local UI

On the remote host:

```bash
cd ml-lab
./scripts/remote_jupyter.sh
```

From the Mac, use a tunnel (replace `GPU_HOST` with an existing SSH host):

```bash
ssh -N -L 8888:127.0.0.1:8888 GPU_HOST
```

Connect Cursor's Jupyter extension to `http://127.0.0.1:8888` using the printed token (Select Kernel -> Existing Jupyter Server). The server stays bound to remote localhost and is not exposed publicly. Remote paths, installed packages, working directory, and datasets belong to the remote machine; only the notebook UI is local.

## Modal (optional, billable)

Do not run this without approval. After approval, install the optional dependency and authenticate:

```bash
uv sync --extra remote
modal token new
MODAL_GPU=T4 modal run remote/modal_example.py
```

The example builds a reproducible image, runs a CUDA matmul, reports GPU information, and is structured for adding a named Modal Volume for checkpoints. No cloud resource is provisioned by this local setup.

## GPU infrastructure notes

For a Linux NVIDIA box, provision and verify the driver first, then choose a compatible CUDA-enabled PyTorch wheel. Install the CUDA toolkit/nvcc only when compiling kernels; install NCCL for distributed training; add Triton/FlashAttention only after checking driver, CUDA, PyTorch, Python, and GPU compatibility. Nsight Systems and Nsight Compute are profiling tools, not Python dependencies. Containerized workflows need an NVIDIA Container Toolkit on the host. None of these Linux-only tools are installed on this Mac.

## Daily reference

```bash
cd ~/Developer/ml-lab
uv sync
uv run python scripts/doctor.py
uv run pytest && uv run ruff check .
uv run python scripts/train.py --device auto --steps 100
```

Use MPS locally with `--device mps`; use CUDA remotely with `--device cuda`. An explicit unavailable device fails instead of silently falling back.

## Verified on this Mac

The local Cursor workbook is `notebooks/03_cursor_workbook.ipynb`. In Cursor, choose `Select Kernel` -> `Jupyter Kernel...` -> `ML Lab (.venv) (Python 3.12.12)`, then choose `Run All`. The workbook was rerun successfully in the Cursor notebook UI: PyTorch 2.11.0 selected Apple MPS, autograd completed, and the inline loss plot rendered.

The CUDA-only workbook is `notebooks/04_remote_cuda_workbook.ipynb`. It intentionally fails rather than falling back when no NVIDIA/CUDA host is attached. This Mac has no `nvidia-smi` or `nvcc`, so remote validation requires setting `GPU_HOST` to a real SSH-accessible Linux GPU box first; no paid GPU instance is currently configured or running.

## Dotfiles and GPU helper

The provider-neutral helper and its instructions live in `dotfiles/README.md`. After copying the example config to `~/.config/ml-lab/gpu.env`, use `gpu doctor`, `gpu connect`, `gpu tunnel`, and `gpu remote-doctor` for an existing SSH host. Provisioning and termination intentionally remain manual so no paid GPU resource is created accidentally. Detailed remote checks are in `remote/README.md`.
