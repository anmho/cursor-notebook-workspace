# Remote GPU provider choices

This workspace keeps the execution interface the same: a Git checkout, `uv`,
PyTorch, and either Cursor Remote SSH or a local Modal function call. The
provider only changes how the remote compute is obtained.

| Provider | Best fit here | Provisioning path | Persistent SSH box |
| --- | --- | --- | --- |
| RunPod | Default remote CUDA workstation | Terraform: `anmho/terraform/projects/cursor-gpu` | Yes |
| Vast.ai | Cheapest marketplace capacity | Official `vastai` CLI/API | Yes |
| Modal | Short remote GPU experiments | Existing `modal_functions.ipynb` | No; function containers |
| Lambda Cloud | More managed GPU VMs | Official Lambda API/CLI | Yes |

## RunPod: Terraform default

Use the existing Terraform root:

```bash
git clone https://github.com/anmho/terraform.git ~/Developer/terraform
cd ~/Developer/terraform
export RUNPOD_API_KEY='YOUR_RUNPOD_API_KEY'
terraform -chdir=projects/cursor-gpu init -reconfigure -backend-config=backend.gcs.hcl
terraform -chdir=projects/cursor-gpu apply
terraform -chdir=projects/cursor-gpu output
```

The root creates one CUDA-enabled PyTorch pod with direct SSH. Register the
public key matching your Cursor SSH identity in RunPod first. Use the
`ssh_direct_*` outputs for `~/.ssh/config`, then connect with Remote SSH and
clone your project on the box.

When finished:

```bash
terraform -chdir=projects/cursor-gpu destroy
```

RunPod's official Terraform provider is the reason this is the preferred
infrastructure path. It manages pods, SSH/Jupyter creation flags, storage, and
the pod lifecycle through Terraform.

## Vast.ai: CLI alternative

Vast.ai is a good low-cost marketplace alternative, but its documented
automation path is the official CLI/API rather than a maintained Terraform
root. Use it when price or availability matters more than having the box
declared in the Terraform repository.

```bash
export VAST_API_KEY='YOUR_VAST_API_KEY'
curl -fsSL https://vast.ai/install.sh | bash
vastai search offers --limit 3
vastai create ssh-key "$(cat ~/.ssh/id_ed25519.pub)"
vastai create instance OFFER_ID --image pytorch/pytorch --disk 32 --ssh --direct
vastai show instances
vastai ssh-url INSTANCE_ID
ssh $(vastai ssh-url INSTANCE_ID)
```

After connecting, clone the project and run `uv sync`, `nvidia-smi`, and the
CUDA notebook exactly as described in `notebooks/remote_cuda.ipynb`. Release
the instance explicitly:

```bash
vastai destroy instance INSTANCE_ID
```

## Modal: remote function alternative

Modal is already included as `notebooks/modal_functions.ipynb`. It is not an
SSH workstation: the notebook and local kernel stay on the Mac, and only the
decorated function runs remotely. This is the preferred path for short,
isolated GPU calls, evaluations, and batch experiments.

```bash
cd ~/Developer/cursor-notebook-workspace
uv sync --extra remote
uv run modal setup
```

Then run the Modal notebook. The remote call is billable and requires a valid
payment method; there is no persistent box to destroy.

## Lambda Cloud: managed VM alternative

Lambda is the better alternative when you want a managed Linux GPU VM and are
willing to use its API/CLI instead of the Terraform repository. Its instances
are ordinary remote machines, so the same SSH, Git clone, `uv sync`, and Cursor
Remote SSH steps apply. The exact instance type and region are account- and
availability-dependent; create it with Lambda's documented API/CLI, then add
the returned host to `~/.ssh/config`.

This workspace does not add a fake Lambda Terraform root. The provider's own
documented automation surface is its API/CLI, while RunPod has the maintained
Terraform provider used by `projects/cursor-gpu`.

## What stays the same

There is no file-sync service in any of these workflows. For a remote box:

1. Commit and push the project on the Mac.
2. Clone it on the GPU machine.
3. Run `uv sync` on the GPU machine.
4. Open the remote checkout in Cursor Remote SSH.
5. Select the remote `.venv/bin/python` notebook kernel.
6. Run `notebooks/remote_cuda.ipynb`.

For Modal, keep the notebook local and call the decorated function instead.
