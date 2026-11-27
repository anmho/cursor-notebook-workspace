import sys

import torch


def test_torch_is_installed() -> None:
    # This catches accidentally running pytest with a global Python instead of this project's .venv.
    assert ".venv" in sys.executable
    assert torch.__version__
