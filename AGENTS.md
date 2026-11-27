# ML engineering project instructions

- Use `uv` for dependency management and the project-local `.venv`.
- Use `uv sync`, `uv run ...`, and commit `pyproject.toml` plus `uv.lock`.
- Keep reusable implementations in `src/ml_lab`; notebooks are for experiments, inspection, and visualization.
- Do not automatically run expensive training or remote/cloud workloads.
- Test CPU functionality before GPU execution.
- Never silently substitute CPU for an explicitly requested GPU; use `ml_lab.device.select_device`.
- Do not install CUDA-only packages on macOS. Linux CUDA dependencies belong in the platform-specific uv source configuration or remote setup.
- Preserve Git-friendly source files and keep generated outputs under ignored directories.
- Use pytest and Ruff for validation; keep type annotations useful without adding unnecessary abstractions.
- Prefer clear, educational implementations for fundamental algorithms.
- Require explicit approval before billable cloud execution (including Modal GPU jobs).
- Never print, commit, or place credentials in source files. Use environment variables or an ignored `.env`.
