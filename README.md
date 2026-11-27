# Minimal ML workflow test

This is intentionally a small validation project for three workflows:

1. Local Cursor notebook with PyTorch/MPS.
2. Cursor Remote SSH notebook with PyTorch/CUDA.
3. Local Cursor notebook calling a remote Modal function.

Notebook-specific usage instructions are in [`notebooks/README.md`](notebooks/README.md).

## Exact creation steps

This is the complete sequence used to create a fresh equivalent workspace:

```bash
mkdir ~/Developer/my-ml-test
cd ~/Developer/my-ml-test
uv init --python 3.12
uv add torch numpy matplotlib
uv add --dev jupyter ipykernel pytest ruff
uv add --optional remote modal
mkdir -p notebooks tests
git init
git add .
git commit -m "Initialize minimal ML workflow test"
cursor -r .
```

The result is only a project directory, `pyproject.toml`, `uv.lock`, `.python-version`, `.venv`, `notebooks/`, `tests/`, and Git metadata.

## Local test

```bash
cd ~/Developer/ml-lab
uv sync
cursor -r .
```

Open `notebooks/local_pytorch.ipynb`, select the local `.venv` Jupyter kernel, and click `Run All`.

## Remote SSH test

Provision a Linux/NVIDIA host and add it to `~/.ssh/config`. Then use Cursor's `Remote-SSH: Connect to Host...` command, open the remote checkout, select its `.venv` kernel, and run `notebooks/remote_cuda.ipynb`.

The remote notebook requires CUDA and intentionally refuses CPU/MPS fallback. No file synchronization is used; the whole project is opened remotely.

## Modal test

```bash
cd ~/Developer/ml-lab
uv sync --extra remote
uv run modal setup
```

Open `notebooks/modal_functions.ipynb` with the local kernel and run all cells. Only the decorated function executes on Modal. The final cell is billable and requires Modal authentication.

## Checks

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

