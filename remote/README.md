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

The provider may restrict kernel debugging or performance counters even when CUDA execution works. Treat those as separate checks.

## Cursor Remote SSH

From Cursor: `Remote-SSH: Connect to Host...` -> select `GPU_HOST` -> open the remote `ml-lab` directory. Select the remote `.venv/bin/python` interpreter and the remote Jupyter kernel. The NVIDIA Nsight VS Code Edition recommendation is remote-only: NVIDIA documents Linux x86_64/arm64 hosts and CUDA Toolkit requirements; it is not installed on this Apple Silicon Mac.

## Remote Jupyter through a local Cursor UI

On the remote host:

```bash
PORT=8888 ./scripts/remote_jupyter.sh
```

On the Mac, in another terminal:

```bash
ssh -N -L 8888:127.0.0.1:8888 GPU_HOST
```

In a `.ipynb` file, choose `Select Kernel` -> `Existing Jupyter Server`, enter `http://127.0.0.1:8888`, and paste the printed token. The server remains bound to remote localhost; do not use `--ip=0.0.0.0` or expose the token publicly.

## Persistent experiments

```bash
tmux new -s ml
uv run python scripts/train.py --device cuda --steps 10000
# detach: Ctrl-b, then d
tmux attach -t ml
```

Use Git for source, persistent storage for checkpoints/datasets, and terminate the instance manually when finished. The repository's `dotfiles/gpu/bin/gpu` helper can connect, tunnel, and run a read-only remote doctor; its `up` and `down` commands intentionally refuse to provision or terminate paid resources.
