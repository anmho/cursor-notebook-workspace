import torch


def test_torch_is_installed() -> None:
    assert torch.__version__
