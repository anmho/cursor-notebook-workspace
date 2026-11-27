# Minimal ML workstation

This repository is a small, reproducible example of three ML workflows:

1. A local Cursor notebook running PyTorch on Apple MPS or CPU.
2. A Cursor Remote SSH notebook running PyTorch on a remote NVIDIA GPU with CUDA.
3. A local Cursor notebook that calls one remote Modal GPU function.

There is no automatic file synchronization layer. In Remote SSH mode, Cursor opens the remote checkout directly. In Modal mode, only the decorated function runs remotely.

Detailed instructions are in [notebooks/README.md](notebooks/README.md), and each notebook repeats the instructions it needs in its first markdown cell.

## What is installed

- Cursor with Python, Jupyter, and Remote SSH extensions.
- uv for the project-local Python environment.
- PyTorch, NumPy, and Matplotlib.
- Jupyter and ipykernel for notebooks.
- pytest and Ruff for checks.
- Optional Modal support in the remote dependency group.

The environment lives at .venv/. It is intentionally ignored by Git and recreated from pyproject.toml and uv.lock.

## Existing workspace: start to finish

Run this on the Mac:

```bash
cd ~/Developer/ml-lab
uv sync
uv run pytest
uv run ruff check .
cursor -r .
```

uv sync creates or reconciles ./.venv from the lockfile. It does not start a notebook kernel.

In Cursor:

1. Open notebooks/local_pytorch.ipynb from the Explorer.
2. In the notebook toolbar, click Select Kernel (or the current kernel name).
3. Choose Select Another Kernel....
4. Choose Python Environments....
5. Choose the workspace interpreter .venv/bin/python.
6. Click Run All.

Selecting the interpreter chooses the kernel. Running the first code cell starts the kernel process on demand. Confirm that the toolbar shows a kernel name, the cells receive execution numbers such as [1], and the first output reports a Python executable inside .venv.

Do not use the Command Palette command Jupyter: Manage Access To Jupyter Kernels; that manages extension permissions and does not select the notebook kernel.

## Create a fresh equivalent workspace

This creates a new independent project with the same fundamentals:

```bash
mkdir -p ~/Developer/my-ml-project
cd ~/Developer/my-ml-project
uv init --python 3.12
uv add torch numpy matplotlib
uv add --dev jupyter ipykernel pytest ruff
mkdir -p notebooks tests
git init
uv sync
cursor -r .
```

Then add your own notebook under notebooks/, select .venv/bin/python through the Cursor kernel picker, and run the first cell. The project is self-contained through its source, pyproject.toml, uv.lock, and folder-local .venv.

## Local PyTorch workflow

Open [notebooks/local_pytorch.ipynb](notebooks/local_pytorch.ipynb) and run all cells. It verifies:

- the active Python executable is the project .venv;
- PyTorch imports successfully;
- the selected device is MPS when available, otherwise CPU;
- matrix multiplication works;
- autograd produces gradients; and
- Matplotlib renders inline.

Stop the local kernel with the notebook Restart menu or by closing Cursor. The notebook file and saved outputs remain on disk; live Python variables do not.

## Remote SSH + NVIDIA CUDA workflow

This mode requires a Linux host with direct SSH access, an NVIDIA driver, and CUDA-capable PyTorch. It is the mode for CUDA, Triton, nvcc, profiling, and full remote repository work.

On the Mac, add a host to ~/.ssh/config:

```sshconfig
Host gpu-box
    HostName YOUR_GPU_HOST
    User YOUR_REMOTE_USER
    IdentityFile ~/.ssh/id_ed25519
```

Test the connection before opening Cursor:

```bash
ssh gpu-box
nvidia-smi
```

On the remote machine, create or clone the project checkout and install its environment:

```bash
cd ~/Developer
git clone YOUR_REPOSITORY_URL ml-lab
cd ~/Developer/ml-lab
uv sync
nvidia-smi
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

If uv is not installed on the remote host, install it there using the official installer or your system package manager, then rerun uv sync.

In Cursor:

1. Run Remote-SSH: Connect to Host....
2. Choose gpu-box.
3. Open the remote ~/Developer/ml-lab folder.
4. Open notebooks/remote_cuda.ipynb.
5. Select the remote .venv/bin/python through Select Another Kernel... -> Python Environments....
6. Click Run All.

The notebook refuses CPU/MPS fallback. If its first cell fails, inspect nvidia-smi, the remote Python path, and torch.cuda.is_available().

When finished, save and commit code, close the remote notebook, disconnect Remote SSH, and stop or terminate the GPU instance in the provider dashboard. Stopping the Cursor window does not stop a rented GPU.

## Modal function-calling workflow

Modal keeps the notebook and its local kernel on the Mac. Only a decorated function is dispatched to a remote GPU.

Prepare the local project:

```bash
cd ~/Developer/ml-lab
uv sync --extra remote
uv run modal setup
```

modal setup authenticates the Modal CLI. Credentials are account configuration and necessarily live outside the repository; never commit them.

Open notebooks/modal_functions.ipynb, select the local .venv/bin/python environment, and run cells in order:

1. Load the Modal client locally.
2. Define the remote GPU function locally.
3. Call remote_matmul.remote(...).

The final cell is billable and requires authentication. It returns the remote GPU name, CUDA device, and tensor shape. The function container is released after the call according to Modal's runtime behavior; there is no persistent notebook server to shut down.

## Verify the repository

```bash
cd ~/Developer/ml-lab
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

## Troubleshooting

### Cursor shows the wrong kernel commands

Use the notebook toolbar's kernel button. The correct path is:

```text
current kernel name / Select Kernel
  -> Select Another Kernel...
  -> Python Environments...
  -> .venv/bin/python
```

Jupyter: Manage Access To Jupyter Kernels is a different Command Palette action.

### The kernel starts with the wrong Python

Check the first cell's sys.executable. It should contain the project path and .venv. Reopen the folder with cursor -r ., then select .venv/bin/python again.

### The remote notebook says CUDA is unavailable

Run these commands in the remote Cursor terminal:

```bash
nvidia-smi
which python
uv run python -c "import sys, torch; print(sys.executable); print(torch.version.cuda); print(torch.cuda.is_available())"
```

### The Modal notebook fails authentication

Run uv run modal setup locally and rerun the notebook's first cell. Do not put Modal credentials in the repository.
