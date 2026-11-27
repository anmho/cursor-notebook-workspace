# Cursor notebook workspace example

This repository is a small, reproducible workspace example for setting up and
using Cursor notebooks with project-local Python environments. It demonstrates:

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

## Cursor configuration included in this repository

The `.vscode/` folder contains optional workspace conveniences that belong with
this example:

- `settings.json` tells the Python extension and new terminals to prefer the
  project-local `.venv/bin/python`.
- `extensions.json` recommends Python, Jupyter, Ruff, and Remote SSH. It does
  not install extensions.

These files do not start a kernel or force a notebook kernel selection. In a
fresh Cursor window, select the notebook kernel explicitly:

1. Click the notebook toolbar's current kernel button.
2. Choose `Select Another Kernel...`.
3. Choose `Python Environments...`.
4. Choose `.venv/bin/python`.

After that, clicking `Run All` or executing the first cell starts the kernel
process on demand. Cursor may remember the selection in that notebook's
metadata, but the repository settings alone do not guarantee that behavior.

This repository does not define notebook keybindings. In the currently tested
Cursor setup, the native labels are `Control+Enter` to execute the current cell
and `Command+Enter` to add a code cell. If those keys are overridden or focus
is elsewhere, use the cell run button or the notebook toolbar's `Run All`.

The global Cursor files in your home directory (`settings.json` and
`keybindings.json`) are personal application settings. They are not copied into
this repository; the current machine's snapshot is kept separately from the
project.

### What had to change for this example

Required for a fresh clone:

1. `pyproject.toml` declares the project name, Python version, and packages.
2. `uv.lock` records the exact resolved dependency versions.
3. The README and notebook instructions explain how to select the project
   interpreter as the notebook kernel.

Conveniences, not requirements:

- `.vscode/settings.json` and `.vscode/extensions.json`.
- No global Cursor keybinding change.
- No global Jupyter kernel registration.
- No copied `.venv`; `uv sync` recreates it locally.
- No file-sync service for Remote SSH; Cursor opens the remote checkout.

## Existing workspace: start to finish

Run this on the Mac:

```bash
cd ~/Developer/cursor-notebook-workspace
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

### Keyboard controls on macOS

In Cursor's notebook UI, `Control+Enter` (`⌃⏎`) executes the current cell. `Command+Enter` (`⌘⏎`) adds a new code cell. To execute the whole notebook, click `Run All` in the notebook toolbar.

If `pydevd` reports that computing a tensor repr was slow, that is debugger inspection overhead rather than a failed cell. The smoke-test tensors are intentionally small; confirm success from the cell execution number and output.

## Create a fresh example workspace from scratch

This is the exact sequence for creating a new independent ML notebook repository:

### 1. Create the folder

```bash
mkdir -p ~/Developer/my-ml-project
cd ~/Developer/my-ml-project
```

`mkdir` creates the folder. `cd` enters it.

### 2. Create project metadata

```bash
uv init --python 3.12
```

This creates `pyproject.toml` and sets the project Python requirement to 3.12.

### 3. Add dependencies

```bash
uv add torch numpy matplotlib
uv add --dev jupyter ipykernel pytest ruff
```

`torch`, `numpy`, and `matplotlib` are runtime dependencies. `jupyter`, `ipykernel`, `pytest`, and `ruff` support notebooks and development checks.

### 4. Create project folders and Git

```bash
mkdir -p notebooks tests
git init
```

### 5. Create the local environment

```bash
uv sync
```

This creates `my-ml-project/.venv/` and installs the locked dependencies. It does not start a kernel.

### 6. Open the project in Cursor

```bash
cursor -r .
```

If the `cursor` command is unavailable, open Cursor manually and choose **File -> Open Folder**.

Install these Cursor extensions if needed:

- Python (`ms-python.python`)
- Jupyter (`ms-toolsai.jupyter`)
- Remote - SSH (`anysphere.remote-ssh`)

### 7. Create and select a notebook kernel

In Cursor:

1. Open the `notebooks` folder.
2. Create `hello.ipynb`.
3. Click the notebook's kernel button.
4. Choose `Select Another Kernel...`.
5. Choose `Python Environments...`.
6. Choose `.venv/bin/python`.

Do not run `ipykernel install --user`; that creates a global kernel registration. The folder-local `.venv/bin/python` is enough.

### 8. Run a test cell

Put this in the first code cell:

```python
import sys

import torch

print("Python:", sys.executable)
print("PyTorch:", torch.__version__)

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print("Device:", device)
```

Click inside the cell and press `Control+Enter` (`⌃⏎`) to run it. `Command+Enter` (`⌘⏎`) adds a code cell. Click **Run All** to execute the entire notebook.

The printed Python path should contain `my-ml-project/.venv/bin/python`. Running the first cell starts the kernel process on demand.

This recipe creates a blank project skeleton. It does not copy this example's
notebooks or `.vscode/` conveniences; create the notebook in the previous step
and copy any project-specific configuration you want.

### 9. Save the new repository

```bash
git add .
git commit -m "Initialize ML project"
```

Each project is independent and contains its own `pyproject.toml`, `uv.lock`, `notebooks/`, `tests/`, and Git history. Running `uv sync` creates the ignored `.venv/` directory locally; it is not committed.

If you physically move an existing checkout to a new folder name, run
`uv sync --reinstall` once. Environment entry-point scripts can contain the
old absolute folder path after a move; a fresh clone only needs ordinary
`uv sync`.

## Local PyTorch workflow

Open [notebooks/local_pytorch.ipynb](notebooks/local_pytorch.ipynb) and run all cells. It verifies:

- the active Python executable is the project .venv;
- PyTorch imports successfully;
- the selected device is MPS when available, otherwise CPU on the Mac;
- matrix multiplication works;
- autograd produces gradients; and
- Matplotlib renders inline.

Restart resets the local kernel state by replacing the running process. Close Cursor or explicitly shut down the kernel to end it. The notebook file and saved outputs remain on disk; live Python variables do not.

## Remote SSH + NVIDIA CUDA workflow

This mode requires a Linux host with direct SSH access, an NVIDIA driver, and CUDA-capable PyTorch. It is the mode for CUDA, Triton, nvcc, profiling, and full remote repository work; those host tools are separate from the Python dependencies installed by this example.

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
git clone YOUR_REPOSITORY_URL cursor-notebook-workspace
cd ~/Developer/cursor-notebook-workspace
uv sync
nvidia-smi
uv run python -c "import torch; available = torch.cuda.is_available(); print({'cuda': available, 'device': torch.cuda.get_device_name(0) if available else None})"
```

If uv is not installed on the remote host, install it there using the official installer or your system package manager, then rerun uv sync.

In Cursor:

1. Run Remote-SSH: Connect to Host....
2. Choose gpu-box.
3. Open the remote ~/Developer/cursor-notebook-workspace folder.
4. Open notebooks/remote_cuda.ipynb.
5. Select the remote .venv/bin/python through Select Another Kernel... -> Python Environments....
6. Click Run All.

The notebook refuses CPU/MPS fallback. If its first cell fails, inspect nvidia-smi, the remote Python path, and torch.cuda.is_available().

When finished, save and commit code, close the remote notebook, disconnect Remote SSH, and stop or terminate the GPU instance in the provider dashboard. Stopping the Cursor window does not stop a rented GPU.

## Modal function-calling workflow

Modal keeps the notebook and its local kernel on the Mac. Only a decorated function is dispatched to a remote GPU.

Prepare the local project:

```bash
cd ~/Developer/cursor-notebook-workspace
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
cd ~/Developer/cursor-notebook-workspace
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
