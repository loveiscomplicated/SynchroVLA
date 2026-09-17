# Project Agent Instructions

## Device Compatibility Contract

All new training, evaluation, and model code must support:

- `cuda`
- `mps`
- `cpu`

Do not hard-code backend-specific calls such as `.cuda()`, `.mps()`, or direct `torch.device("cuda")` / `torch.device("mps")` in task code. Use the shared device utility:

```python
from vla_gnn_recurrent.utils import select_device

device = select_device("auto")
model = model.to(device)
tensor = tensor.to(device)
```

Default `auto` order is:

1. CUDA if available
2. MPS if available
3. CPU

MuJoCo physics remains CPU unless a future task explicitly ports the simulator. CUDA/MPS currently accelerate PyTorch model training/inference only.

## Training vs Rollout Devices

Closed-loop MuJoCo rollout code should keep these concepts separate:

- training/model device, e.g. `--device cuda`
- rollout inference device, e.g. `--eval-device cpu`
- simulator execution, currently CPU MuJoCo

For episode-parallel rollout:

- use spawn-safe multiprocessing
- never share a MuJoCo environment across workers
- each worker must create its own environment
- prefer loading a checkpoint inside the worker instead of passing a live model object
- use fixed per-episode seeds
- return results sorted by episode index

## CLI Standard

New train/eval scripts should expose:

- `--device {auto,cpu,mps,cuda}`
- `--eval-device {auto,cpu,mps,cuda}` when rollout/evaluation can differ from training
- `--eval-workers` when closed-loop episodes can be parallelized

## Pre-Final Checklist

Before finishing a model/training/evaluation change:

- no hard-coded backend-specific device calls
- checkpoint load uses `map_location`
- CPU path still works
- MPS path still works where available
- CUDA path is supported or explicitly documented
- closed-loop evaluation does not call scripted experts
- worker rollouts do not share MuJoCo environments
