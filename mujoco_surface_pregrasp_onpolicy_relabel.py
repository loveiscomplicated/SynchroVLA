from __future__ import annotations

import argparse

from vla_gnn_recurrent.training.surface_pregrasp_onpolicy_relabel import (
    DEFAULT_OUTPUT,
    DEFAULT_UPDATES,
    SEEDS,
    run_experiment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect and oracle-relabel visited Surface Set states, then compare equal-budget FF continuations."
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--updates", type=int, default=DEFAULT_UPDATES)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--eval-device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--eval-workers", type=int, default=1)
    parser.add_argument("--latency-warmup", type=int, default=200)
    parser.add_argument("--latency-iterations", type=int, default=2000)
    args = parser.parse_args()
    if args.eval_workers != 1:
        parser.error("This experiment runs sequentially; each rollout owns its MuJoCo environment. Use --eval-workers 1.")
    if args.updates < 1:
        parser.error("--updates must be positive")
    if args.latency_warmup < 200 or args.latency_iterations < 2000:
        parser.error("Latency validation requires at least 200 warm-up and 2000 timed iterations.")
    return args


def main() -> None:
    args = parse_args()
    result = run_experiment(
        output_dir=args.output_dir,
        seeds=tuple(args.seeds),
        updates=args.updates,
        device_preference=args.device,
        eval_device_preference=args.eval_device,
        latency_warmup=args.latency_warmup,
        latency_iterations=args.latency_iterations,
    )
    print(f"completed: {args.output_dir}/summary.md")
    print(f"representation reevaluation gate: {result['representation_reevaluation_gate_passed']}")


if __name__ == "__main__":
    main()
