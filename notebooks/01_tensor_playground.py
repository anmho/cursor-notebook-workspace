# %% [markdown]
# # Tensor playground
#
# Run cells from Cursor's Python Interactive window, or run the whole file with
# `uv run python notebooks/01_tensor_playground.py`.

# %%
import os

import matplotlib.pyplot as plt
import torch

from ml_lab.device import device_report, select_device

device = select_device("auto")
print(device_report().as_dict())

# %%
x = torch.linspace(0, 2 * torch.pi, 100, device=device)
y = torch.sin(x)
print(f"device={y.device}, shape={tuple(y.shape)}, dtype={y.dtype}")

# %%
plt.plot(x.cpu(), y.cpu())
plt.title("A tensor computed on the selected device")
plt.xlabel("x")
plt.ylabel("sin(x)")
if os.environ.get("ML_LAB_INTERACTIVE_PLOT") == "1":
    plt.show()
else:
    output = "artifacts/tensor_playground.png"
    plt.savefig(output, dpi=120, bbox_inches="tight")
    print(f"saved plot to {output}")
plt.close()

# %% [markdown]
# Markdown cells are regular comments, so this file remains executable from a terminal.
