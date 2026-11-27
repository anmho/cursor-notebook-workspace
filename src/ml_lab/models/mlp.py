"""A deliberately small MLP used by examples and tests."""

from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim: int = 4, hidden_dim: int = 16, output_dim: int = 1) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.network(x)
