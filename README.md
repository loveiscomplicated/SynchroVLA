# VLA-GNN-Recurrent

Minimal prototype for validating closed-loop graph-conditioned end-effector control.

This repository intentionally implements only:

```text
Synthetic Dynamic Graph -> Edge-aware GNN -> GRU or MLP -> Delta EE Action -> Synthetic Environment Update
```

It does not include VLMs, perception stacks, robot simulation, IK, ROS, CUDA-only dependencies, or PyTorch Geometric.

## Setup

Python 3.11+ is required.

```bash
uv sync --extra dev --python 3.11
```

If you prefer plain pip:

```bash
python3.11 -m pip install -r requirements.txt
```

## Tests

```bash
uv run pytest
```

## Train

Train both the recurrent controller and the feed-forward baseline:

```bash
uv run python train.py --model both --episodes 160 --max-steps 48 \
  --task moving --observation-intervals 1,2,4,8 \
  --action-prior none --no-target-velocity-feature \
  --bptt-steps 4 --sequence-repeats 8 \
  --output-dir artifacts/stale_final/checkpoints
```

The default `--action-prior none` setting is learned-only:

```text
observed graph -> GNN -> MLP or GRU -> learned Delta EE
```

Use `--action-prior geometric` to enable the previous geometric-prior + residual baseline. Training collects sequential teacher-rollout episodes and replays those temporal graph/action sequences with chunked BPTT. GRU hidden state is reset only at episode boundaries and detached only at BPTT chunk boundaries.

## Evaluate

Normal reaching:

```bash
uv run python eval.py --model both --checkpoint-dir artifacts/stale_final/checkpoints \
  --episodes 64 --max-steps 60 --task moving \
  --observation-intervals 1,2,4,8 \
  --no-target-velocity-feature --plot \
  --output-dir artifacts/stale_final/eval_matrix
```

Perturbed target reaching with trajectory plots:

```bash
uv run python eval.py --model both --checkpoint-dir artifacts/stale_final/checkpoints \
  --episodes 32 --max-steps 60 --task static \
  --observation-intervals 1,4 --perturb --perturb-step 10 \
  --no-target-velocity-feature --plot \
  --output-dir artifacts/stale_final/eval_perturb
```

## Architecture

Per timestep, the environment emits an `ObjectObservation` plus end-effector state. `GraphBuilder` converts those states into a typed directed graph:

- Node features: `[xyz(3), velocity(3), node_type_one_hot(4), task_flag(1)]`, shape `[num_nodes, 11]`
- Edge features: `[relative_position(3), distance(1), edge_type_one_hot(3)]`, shape `[num_edges, 7]`
- Spatial edges: `end_effector <-> object`
- Task edges: `task <-> end_effector`, `task <-> object`
- Optional dummy kinematic edges can be added through `GraphBuilder(include_joints=True)`

The graph encoder is a 3-layer edge-aware message passing network with hidden size 128. Each message MLP receives both source node embeddings and edge features. The final control readout combines `[EE embedding, object embedding, global mean]` with a learned linear geometry readout of observed `[relative xyz, distance]`, returning shape `[1, 128]`. This geometry readout is learned representation, not a manually constructed action.

The recurrent controller feeds the graph readout through a 2-layer GRU with hidden size 256. The feed-forward baseline uses the same graph encoder and an MLP action head. Both controllers support:

- `--action-prior none`: learned-only bounded action
- `--action-prior geometric`: `clamp(geometric_graph_prior + learned_residual)`

For stale observation experiments, `StaleTargetObserver` separates true environment state from controller-observed object state. The end-effector state updates every timestep, while the object node re-anchors only every `N` control steps. With `--no-target-velocity-feature`, target velocity is zeroed in object node features.

## Latest Stale-Observation Result

Moving target, learned-only, no target velocity feature, 64 eval episodes:

| Model | N=1 success / final | N=2 success / final | N=4 success / final | N=8 success / final |
| --- | ---: | ---: | ---: | ---: |
| Feed-forward | 0.812 / 0.044 | 0.781 / 0.046 | 0.641 / 0.055 | 0.516 / 0.060 |
| GRU | 0.922 / 0.043 | 0.906 / 0.044 | 0.750 / 0.051 | 0.578 / 0.065 |

The GRU has higher success rate and smoother actions across all stale intervals in this run, while final distance is similar at small N and worse for GRU at N=8.

## Scope Boundary

The synthetic object state enters only through `ObjectObservation`. Later, a perception stack can produce the same interface:

```text
GroundingDINO/SAM2/RGB-D -> ObjectObservation -> GraphBuilder
```

No changes should be needed in the GNN or controllers.
