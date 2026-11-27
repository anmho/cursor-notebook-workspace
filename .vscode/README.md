# Workspace Cursor configuration

These files are workspace conveniences, not a replacement for the notebook
kernel picker.

## `settings.json`

- `python.defaultInterpreterPath` points Cursor's Python extension at
  `.venv/bin/python` inside this repository.
- `python.terminal.activateEnvironment` activates that environment in new
  integrated terminals.
- `jupyter.askForKernelRestart` controls whether Cursor asks before restarting
  a Jupyter kernel; it does not select a kernel.
- `notebook.lineNumbers` keeps notebook line numbers visible.

The first time you open a notebook, select its kernel manually:

```text
Current kernel -> Select Another Kernel...
  -> Python Environments...
  -> .venv/bin/python
```

Running the first cell starts that Python kernel process. There is no
`ipykernel install --user` step and no separate kernel file stored here.

## `extensions.json`

This only recommends the Python, Jupyter, Ruff, and Remote SSH extensions. It
does not install them.

There are no workspace keybindings here. Cursor's tested native notebook
shortcuts are `Control+Enter` to execute a cell and `Command+Enter` to add a
code cell.
