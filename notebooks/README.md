# Notebook workflows

These are three separate ways to use the same minimal project. Open the folder in Cursor first:

```bash
cd ~/Developer/ml-lab
uv sync
cursor -r .
```

In Cursor, open a notebook by selecting it in the Explorer. Use the kernel picker in the upper-right of the notebook editor — not the Command Palette search.

![Cursor notebook kernel picker flow](../docs/images/cursor-kernel-picker-flow.svg)

The hermetic flow is: click the current kernel button (`Select Kernel` on first use, or the current kernel name) -> `Select Another Kernel...` -> `Python Environments...` -> choose `.venv/bin/python`. The Command Palette's `Jupyter: Manage Access To Jupyter Kernels` result does not select the notebook kernel.

## 1. Local PyTorch: `local_pytorch.ipynb`

Purpose: run ordinary PyTorch experiments on this Mac.

1. Open [`local_pytorch.ipynb`](local_pytorch.ipynb).
2. In the notebook toolbar above the first cell, click `Select Kernel` (or the current kernel name).
3. Choose `Select Another Kernel...`, then `Python Environments...`, then choose the project `.venv/bin/python` environment.
4. Click `Run All` in that same notebook toolbar.

The notebook prints the Python/PyTorch/device information, performs a matrix multiplication, runs autograd, and renders a plot. On this Mac it should select Apple MPS; CPU is the fallback.

## 2. Cursor Remote SSH + CUDA: `remote_cuda.ipynb`

Purpose: develop and execute the entire project on a Linux/NVIDIA machine. Notebook files and the Python environment are remote in this mode; there is no file-sync layer.

Prerequisite: add a real GPU host to `~/.ssh/config`, then verify it in a terminal:

```bash
ssh gpu-box
```

On the remote machine, from the project checkout:

```bash
uv sync
nvidia-smi
```

In Cursor:

1. Run `Remote-SSH: Connect to Host...`.
2. Choose `gpu-box`.
3. Open the remote `ml-lab` folder.
4. Open [`remote_cuda.ipynb`](remote_cuda.ipynb).
5. In the notebook toolbar above the first cell, click `Select Kernel` (or the current kernel name).
6. Choose `Select Another Kernel...`, then `Python Environments...`, then choose the remote `.venv/bin/python` environment.
7. Click `Run All` in that same notebook toolbar.

The first cell prints the CUDA version and GPU name. The notebook intentionally raises an error if CUDA is unavailable; it never falls back to Mac MPS or CPU.

## 3. Modal function calling: `modal_functions.ipynb`

Purpose: keep the notebook local while dispatching one selected function to a remote Modal GPU.

Prepare the local project once:

```bash
uv sync --extra remote
uv run modal setup
```

Then:

1. Keep Cursor connected to the local folder.
2. Open [`modal_functions.ipynb`](modal_functions.ipynb).
3. In the notebook toolbar above the first cell, click `Select Kernel` (or the current kernel name).
4. Choose `Select Another Kernel...`, then `Python Environments...`, then choose the local `.venv/bin/python` environment.
5. Run the first two cells. They load Modal locally and define the remote function.
6. Run the final cell. It calls `remote_matmul.remote(...)` on a Modal T4 GPU.

The final cell is a billable cloud operation and requires Modal authentication. The notebook file and local source remain on the Mac; only the decorated function runs remotely.

## Which one should I use?

| Notebook | Execution | Best for |
| --- | --- | --- |
| `local_pytorch.ipynb` | Mac MPS/CPU | Learning and ordinary experiments |
| `remote_cuda.ipynb` | Full remote Linux/NVIDIA project | CUDA, Triton, `nvcc`, profiling |
| `modal_functions.ipynb` | Local notebook + selected remote function | Isolated GPU experiments |

