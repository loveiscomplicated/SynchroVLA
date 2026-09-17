# Vast.ai CUDA Setup for SynchroVLA

This guide sets up a Vast.ai NVIDIA/CUDA machine for the static Pick-and-Place 2x2 convergence experiment.

The intended execution model is:

```text
CUDA:
  PyTorch training

CPU workers:
  MuJoCo closed-loop rollout episodes

Git:
  source code only

Manual artifact transfer:
  demonstration datasets and optional previous experiment outputs
```

MuJoCo physics is still CPU-based. CUDA accelerates PyTorch model training/inference, while `--eval-workers` parallelizes closed-loop MuJoCo episodes across CPU processes.

## 1. Choose a Vast.ai Image

Prefer a template with:

- NVIDIA driver already working
- CUDA runtime installed
- Python 3.11+
- PyTorch CUDA preinstalled if available

Useful first checks after SSH:

```bash
nvidia-smi
python --version
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

If `torch.cuda.is_available()` is already `True`, keep the existing PyTorch install unless there is a specific version conflict.

Official PyTorch installation selector:

<https://pytorch.org/get-started/locally/>

## 2. SSH Into the Instance

Example:

```bash
ssh -p 40181 root@181.99.244.111 -L 8080:localhost:8080
```

The `-L 8080:localhost:8080` part is only for port forwarding. It is not needed for file transfer.

## 3. Clone the Repository

On Vast.ai:

```bash
cd /workspace
git clone https://github.com/loveiscomplicated/SynchroVLA.git
cd SynchroVLA
```

If the repository already exists:

```bash
cd /workspace/SynchroVLA
git pull origin main
```

## 4. Python Environment

If the image already provides a CUDA-enabled Python environment, you can use it directly.

If you need a clean environment:

```bash
conda create -n synchrovla python=3.11 -y
conda activate synchrovla
python -m pip install --upgrade pip
```

If PyTorch CUDA is missing, install the CUDA wheel matching the PyTorch selector. Example for CUDA 12.6:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

Then install project dependencies:

```bash
pip install -r requirements.txt
```

Verify:

```bash
python - <<'PY'
import torch
import mujoco
import dm_control
print("torch", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("mujoco", mujoco.__version__)
print("dm_control ok")
PY
```

## 5. MuJoCo Headless Settings

For headless rendering/simulation, set:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

Optional: persist these in the shell:

```bash
cat >> ~/.bashrc <<'EOF'
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
EOF
source ~/.bashrc
```

If MuJoCo complains about missing GL/EGL libraries, install system packages:

```bash
apt-get update
apt-get install -y git rsync libgl1 libegl1 libgles2 libosmesa6 libglfw3 patchelf
```

## 6. Transfer Required Dataset Artifacts

The convergence experiment expects this dataset:

```text
artifacts/mujoco_pick_place_alignment/demos/pick_place_alignment_dagger.pt
artifacts/mujoco_pick_place_alignment/demos/pick_place_alignment_dagger_metadata.json
```

These files are intentionally not committed because `artifacts/` is gitignored.

From the **Mac/local repo root**, create the remote directory:

```bash
ssh -p 40181 root@181.99.244.111 \
  'mkdir -p /workspace/SynchroVLA/artifacts/mujoco_pick_place_alignment/demos'
```

Then transfer:

```bash
rsync -avP -e "ssh -p 40181" \
  artifacts/mujoco_pick_place_alignment/demos/pick_place_alignment_dagger.pt \
  artifacts/mujoco_pick_place_alignment/demos/pick_place_alignment_dagger_metadata.json \
  root@181.99.244.111:/workspace/SynchroVLA/artifacts/mujoco_pick_place_alignment/demos/
```

On Vast.ai, verify:

```bash
cd /workspace/SynchroVLA
ls -lh artifacts/mujoco_pick_place_alignment/demos/
```

Expected:

```text
pick_place_alignment_dagger.pt
pick_place_alignment_dagger_metadata.json
```

Optional, for fixed-budget comparison summaries:

```bash
rsync -avP -e "ssh -p 40181" \
  artifacts/static_2x2_ablation/ \
  root@181.99.244.111:/workspace/SynchroVLA/artifacts/static_2x2_ablation/
```

The fixed-budget artifacts are not required for training. They are only needed if you want the final summary to compare fixed-budget vs convergence-aware results.

## 7. Run Tests

Fast compatibility checks:

```bash
python -m pytest tests/test_device_contract.py tests/test_static_2x2_convergence.py -q
```

Full test suite:

```bash
python -m pytest -q
```

## 8. Run the Convergence-Aware Static 2x2 Experiment

Recommended Vast.ai command:

```bash
python scripts/run_and_wait.py \
  --stdout-log artifacts/static_2x2_convergence/orchestrator.stdout.log \
  --stderr-log artifacts/static_2x2_convergence/orchestrator.stderr.log \
  --metadata-path artifacts/static_2x2_convergence/orchestrator_metadata.json \
  --stream-output \
  -- python mujoco_static_2x2_convergence.py run-all \
    --device cuda \
    --eval-device cpu \
    --eval-workers 8 \
    --stream-child-output
```

Meaning:

```text
--device cuda
  Train PyTorch models on CUDA.

--eval-device cpu
  Run policy inference inside MuJoCo rollout workers on CPU.
  This avoids multiple small worker processes competing for one GPU.

--eval-workers 8
  Run closed-loop validation/test episodes in parallel across CPU workers.

--stream-child-output
  Show tqdm progress bars from child runs in the terminal.
```

If the instance has fewer CPU cores, use fewer rollout workers:

```bash
--eval-workers 4
```

## 9. Resume Behavior

The orchestrator is resumable.

Before launching each model/seed run, it checks:

```text
artifacts/static_2x2_convergence/runs/<model>_seed<seed>/run_result.json
```

If the run is complete and the best checkpoint exists, it skips that run.

If a run failed before completion, re-running the same command restarts only that incomplete run.

To manually clear one failed run:

```bash
rm -rf artifacts/static_2x2_convergence/runs/flat_ff_seed1601
```

Then rerun the same command.

## 10. Summarize Existing Results

If all runs are complete:

```bash
python mujoco_static_2x2_convergence.py summarize \
  --output-root artifacts/static_2x2_convergence \
  --fixed-budget-root artifacts/static_2x2_ablation
```

Main outputs:

```text
artifacts/static_2x2_convergence/summary/summary.json
artifacts/static_2x2_convergence/summary/final_metrics_by_seed.csv
artifacts/static_2x2_convergence/summary/factorial_effects.json
artifacts/static_2x2_convergence/summary/fixed_budget_vs_convergence.json
artifacts/static_2x2_convergence/summary/aggregate_convergence_curves.png
```

## 11. Common Failure: Missing Dataset

Error:

```text
FileNotFoundError:
artifacts/mujoco_pick_place_alignment/demos/pick_place_alignment_dagger.pt
```

Cause:

```text
artifacts/ is gitignored, so the dataset was not cloned from GitHub.
```

Fix:

```bash
rsync -avP -e "ssh -p 40181" \
  artifacts/mujoco_pick_place_alignment/demos/pick_place_alignment_dagger.pt \
  artifacts/mujoco_pick_place_alignment/demos/pick_place_alignment_dagger_metadata.json \
  root@181.99.244.111:/workspace/SynchroVLA/artifacts/mujoco_pick_place_alignment/demos/
```

## 12. Common Failure: CUDA Not Available

Check:

```bash
nvidia-smi
python - <<'PY'
import torch
print(torch.cuda.is_available())
print(torch.version.cuda)
PY
```

If CUDA is unavailable:

1. Confirm the Vast.ai instance has an NVIDIA GPU.
2. Confirm the Docker image has NVIDIA runtime access.
3. Install a CUDA-enabled PyTorch wheel using the official selector.

## 13. Common Failure: MuJoCo EGL/GL Error

Set:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

Install missing libraries if needed:

```bash
apt-get update
apt-get install -y libgl1 libegl1 libgles2 libosmesa6 libglfw3 patchelf
```

## 14. Notes on Performance

CUDA does not make current MuJoCo physics run on GPU.

Expected acceleration comes from:

```text
1. CUDA training for PyTorch models
2. CPU-parallel MuJoCo closed-loop validation/test episodes
```

The default recommended split is:

```text
Training:
  --device cuda

Rollout:
  --eval-device cpu
  --eval-workers 4 or 8
```

Only consider `--eval-device cuda` later if the policy/perception model becomes large enough that rollout inference dominates CPU simulation overhead.
