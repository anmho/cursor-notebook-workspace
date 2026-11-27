# Remote NVIDIA workflow

This project is designed so the Mac remains the editor and local CPU/MPS machine, while a disposable Linux/NVIDIA host provides CUDA, `nvcc`, Triton, CuPy, and profiling tools when needed.

## Provisioning checklist

Use an approved provider account and provision manually so billing is visible. Require:

- SSH access and a Linux CUDA development image, not only a runtime image.
- NVIDIA driver plus CUDA Toolkit (`nvidia-smi`, `nvcc`).
- Persistent disk for the checkout, environments, datasets, and checkpoints.
- Permission for `cuda-gdb`, Compute Sanitizer, Nsight Systems, and Nsight Compute if profiling matters.
- A host key you can verify before adding it to `~/.ssh/known_hosts`.

Add the host to `~/.ssh/config`; do not put passwords, API keys, or temporary provider tokens in this repository.

## Bootstrap and verify

```bash
ssh GPU_HOST
git clone <repository-url> ml-lab
cd ml-lab
./scripts/bootstrap.sh
uv sync --extra gpu
nvidia-smi
nvcc --version
uv run python scripts/doctor.py
```

Then run the small examples:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build
uv run python kernels/cupy_square.py
uv run python kernels/triton_add.py
```

Open `notebooks/04_remote_cuda_workbook.ipynb` in the remote Cursor window and select the remote `.venv` kernel. It asserts `torch.cuda.is_available()` and uses `torch.device("cuda")`; it never silently falls back to CPU or MPS.

The provider may restrict kernel debugging or performance counters even when CUDA execution works. Treat those as separate checks.

## Cursor Remote SSH

From Cursor: `Remote-SSH: Connect to Host...` -> select `GPU_HOST` -> open the remote `ml-lab` directory. Select the remote `.venv/bin/python` interpreter and the remote Jupyter kernel. The NVIDIA Nsight VS Code Edition recommendation is remote-only: NVIDIA documents Linux x86_64/arm64 hosts and CUDA Toolkit requirements; it is not installed on this Apple Silicon Mac.

For a fundamentals-first setup, add a real host manually using [`ssh-config.example`](ssh-config.example), then use ordinary SSH commands:

```bash
ssh gpu-box
cd ~/Developer/ml-lab
uv sync
nvidia-smi
```

The optional local `gpu` command only wraps basic connection and read-only doctor commands. It does not provision machines, clone repositories, synchronize files, or manage Modal.

## Persistent experiments

```bash
tmux new -s ml
uv run python scripts/train.py --device cuda --steps 10000
# detach: Ctrl-b, then d
tmux attach -t ml
```

Use Git for source, persistent storage for checkpoints/datasets, and terminate the instance manually when finished. The separate `~/Developer/ml-workstation` helper can connect and run a read-only remote doctor; its `up` and `down` commands intentionally refuse to provision or terminate paid resources.
