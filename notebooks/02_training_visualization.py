"""Jupyter-friendly training visualization; keep expensive work explicit."""

# %% [markdown]
# # Training visualization
#
# Run cells in Cursor with the Jupyter extension. The training cell is explicit,
# so editing visualization cells does not silently restart a long experiment.

# %%
import os

import matplotlib.pyplot as plt
import torch

from ml_lab.device import select_device
from ml_lab.models.mlp import MLP

device = select_device("auto")
steps = 50

# %%
torch.manual_seed(7)
model = MLP().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
inputs = torch.randn(32, 4, device=device)
target = inputs.sum(dim=1, keepdim=True)
losses = []

for _ in range(steps):
    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.mse_loss(model(inputs), target)
    loss.backward()
    optimizer.step()
    losses.append(float(loss.detach().cpu()))

print({"device": str(device), "final_loss": losses[-1]})

# %%
plt.plot(losses)
plt.title(f"Training loss ({device})")
plt.xlabel("step")
plt.ylabel("MSE")
if os.environ.get("ML_LAB_INTERACTIVE_PLOT") == "1":
    plt.show()
else:
    plt.savefig("artifacts/training_visualization.png", dpi=120, bbox_inches="tight")
    print("saved plot to artifacts/training_visualization.png")
plt.close()
