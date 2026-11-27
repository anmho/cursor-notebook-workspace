# Notebook runbooks

This folder contains three complete workflows. They share the same project dependency files but have different execution locations:

| Notebook | Kernel/function location | Use it for |
| --- | --- | --- |
| local_pytorch.ipynb | Mac .venv kernel | Learning, MPS, CPU |
| remote_cuda.ipynb | Remote Linux .venv kernel | CUDA, Triton, nvcc, profiling |
| modal_functions.ipynb | Local kernel; remote function container | Discrete remote GPU calls |

## One-time Cursor setup

Install these Cursor extensions:

- Python (ms-python.python)
- Jupyter (ms-toolsai.jupyter)
- Ruff (charliermarsh.ruff)
- Remote - SSH (anysphere.remote-ssh)

Open the project:

```bash
cd ~/Developer/cursor-notebook-workspace
uv sync
cursor -r .
```

![Cursor notebook kernel picker flow](../docs/images/cursor-kernel-picker-flow.svg)

The hermetic kernel flow is:

```text
current kernel name / Select Kernel
  -> Select Another Kernel...
  -> Python Environments...
  -> .venv/bin/python
```

The kernel starts automatically when you run the first code cell. There is no separate kernel-start command in the normal Cursor workflow. Confirm startup when the toolbar shows the selected environment, the cell receives an execution number such as [1], and the first output prints sys.executable inside .venv.

Do not use the Command Palette's Jupyter: Manage Access To Jupyter Kernels; it manages extension access and is not the notebook kernel picker.

## macOS keyboard controls

Cursor currently labels the notebook commands as follows:

| Action | Key |
| --- | --- |
| Execute the current cell | `Control+Enter` (`⌃⏎`) |
| Add a code cell | `Command+Enter` (`⌘⏎`) |
| Run every cell | Click `Run All` in the notebook toolbar |
| Restart the kernel | Click `Restart` in the notebook toolbar |

Important: `Command+Enter` is not the run-cell shortcut here; it adds a code cell. Use `Control+Enter` to execute the selected cell.

If Cursor prints `pydevd warning: Computing repr ... was slow`, the cell may still have completed successfully. That is debugger variable inspection timing, not a PyTorch failure. These smoke-test notebooks use small tensors to avoid unnecessary debugger inspection overhead.

## Local PyTorch: local_pytorch.ipynb

### Setup

```bash
cd ~/Developer/cursor-notebook-workspace
uv sync
cursor -r .
```

### Run

1. Open local_pytorch.ipynb from the Explorer.
2. Select the project .venv/bin/python using the kernel flow above.
3. Click Run All.

The first cell starts the local Python kernel and verifies its executable, Python version, PyTorch version, and device. The remaining cells test matrix multiplication, autograd, and inline plotting.

### Stop

Use the notebook Restart menu to replace the running process and reset Python state. Close Cursor or explicitly shut down the kernel to end it. Saved notebook outputs remain in the .ipynb; variables in memory do not.

## Remote SSH + CUDA: remote_cuda.ipynb

### Requirements

- A Linux host with an NVIDIA GPU.
- SSH access from the Mac.
- An NVIDIA driver and a working nvidia-smi command.
- uv installed on the remote host.
- A remote checkout of this repository.

### Provision the box first

This notebook does not create the provider instance. Before opening Cursor,
manually create a Linux GPU instance/Pod in RunPod, Vast, or another provider:

1. Choose an NVIDIA GPU and a PyTorch/CUDA Linux image.
2. Enable direct SSH and a public IP or reachable SSH endpoint.
3. Add the Mac public key from `~/.ssh/id_ed25519.pub`; never upload the private
   key.
4. Attach persistent storage for checkpoints or datasets if needed.
5. Configure an automatic stop/termination limit.
6. Copy the actual SSH host, port, username, and instance status from the
   provider's connection panel.

For a RunPod Pod, use SSH enabled with port `22/tcp` exposed and a public IP
when direct SSH is required. The connection panel or Pod details provide the
real host and port.

Add the host locally:

```sshconfig
Host gpu-box
    HostName YOUR_GPU_HOST
    User YOUR_REMOTE_USER
    IdentityFile ~/.ssh/id_ed25519
```

Test it:

```bash
ssh gpu-box
nvidia-smi
```

### Prepare the remote checkout

Run on the remote host:

```bash
cd ~/Developer
git clone YOUR_REPOSITORY_URL cursor-notebook-workspace
cd ~/Developer/cursor-notebook-workspace
uv sync
nvidia-smi
uv run python -c "import torch; available = torch.cuda.is_available(); print({'cuda': available, 'device': torch.cuda.get_device_name(0) if available else None})"
```

There is no automatic file synchronization. Cursor opens and edits the remote checkout directly.

### Run in Cursor

1. Run Remote-SSH: Connect to Host....
2. Choose gpu-box.
3. Open the remote cursor-notebook-workspace folder.
4. Open remote_cuda.ipynb.
5. Select the remote .venv/bin/python using Select Another Kernel... -> Python Environments....
6. Click Run All.

The first cell starts the remote kernel and requires CUDA. The notebook intentionally fails instead of falling back to CPU or MPS.

### Stop and release resources

Commit or push code, close the notebook, disconnect Remote SSH, and stop or terminate the GPU instance with its provider. Closing Cursor alone does not stop a rented machine.

## Modal function calling: modal_functions.ipynb

### Setup

```bash
cd ~/Developer/cursor-notebook-workspace
uv sync --extra remote
uv run modal setup
```

Authentication is stored outside this repository. Never commit Modal credentials.

### Run

1. Keep Cursor connected to the local cursor-notebook-workspace folder.
2. Open modal_functions.ipynb.
3. Select the local .venv/bin/python using the kernel flow above.
4. Run the first cell: this starts the local notebook kernel and loads Modal.
5. Run the second cell: this defines the remote GPU function locally.
6. Run the final cell: this calls the function on a Modal T4 GPU.

The final call is billable. It returns the remote CUDA device, GPU name, and result shape. Modal handles the remote container lifecycle; there is no persistent remote notebook kernel to shut down.

## Kernel versus server versus function

- A notebook kernel is a running Python process holding cell state in memory.
- A Jupyter server manages notebook connections and starts kernels.
- Cursor normally starts both pieces as needed.
- A Modal function call is a separate remote execution; it is not the same as a persistent notebook kernel.

## Checks

```bash
cd ~/Developer/cursor-notebook-workspace
uv run pytest
uv run ruff check .
uv run ruff format --check .
```
