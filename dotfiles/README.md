# ML lab dotfiles

This directory contains safe, provider-neutral helpers for the local Mac and a future remote NVIDIA box. It does not create, start, stop, or terminate cloud instances, and it does not modify the existing global shell or SSH configuration.

## Install the helper

```bash
mkdir -p ~/.local/bin ~/.config/ml-lab
ln -sf "$HOME/Developer/ml-lab/dotfiles/gpu/bin/gpu" "$HOME/.local/bin/gpu"
cp "$HOME/Developer/ml-lab/dotfiles/gpu/config.env.example" "$HOME/.config/ml-lab/gpu.env"
```

Edit `~/.config/ml-lab/gpu.env` and set `GPU_HOST` to a host already present in `~/.ssh/config`. Keep provider credentials in the provider's own CLI/keychain, not in this file.

## Commands

```bash
gpu doctor          # Check local tools and the configured SSH host
gpu connect         # Open the remote workspace in Cursor
gpu tunnel          # Forward the remote Jupyter server to localhost:8888
gpu remote-doctor   # Read-only CUDA/PyTorch/toolchain probe
gpu up              # Refuses: no billable provisioning without approval
gpu down            # Refuses: no destructive termination without explicit setup
```

For a future rented box, prefer an SSH-capable Linux image with an NVIDIA driver, CUDA toolkit (`nvcc`), persistent storage, and permission for CUDA profiling counters. Containers are adequate for PyTorch/Triton/CuPy/CUDA exercises; use a VM or bare metal if the provider blocks `cuda-gdb`, Compute Sanitizer, Nsight, or custom driver work.

Recommended low-cost workflow: provision manually from an approved Vast.ai, RunPod, Lium, or equivalent account; add the returned host to SSH; set `GPU_HOST`; run `gpu remote-doctor`; then open Cursor with `gpu connect`. Terminate the instance manually when finished. Rates and availability are provider-dependent and are deliberately not hard-coded.

## Remote environment bootstrap

On the remote host, from the project checkout:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.12
uv sync --extra gpu
./scripts/remote_jupyter.sh
```

Use the existing project `README.md` for the SSH tunnel, Jupyter server URL, `tmux`, CUDA/PyTorch compatibility, and profiling notes.
