from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm


MODELS = ("flat_ff", "flat_gru", "graph_ff", "graph_gru")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tail(path: Path, lines: int = 60) -> str:
    if not path.exists():
        return ""
    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(content[-lines:])


def _run_job(job: dict[str, Any]) -> dict[str, Any]:
    stdout_path = Path(job["stdout_log"])
    stderr_path = Path(job["stderr_log"])
    metadata_path = Path(job["metadata_path"])
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(job["gpu"])
    started_at = _utc_now()
    start = time.perf_counter()
    with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open("w", encoding="utf-8") as stderr_file:
        result = subprocess.run(job["command"], stdout=stdout_file, stderr=stderr_file, env=env, check=False)
    payload = {
        "model": job["model"],
        "seed": job["seed"],
        "gpu": job["gpu"],
        "command": job["command"],
        "exit_code": int(result.returncode),
        "started_at": started_at,
        "finished_at": _utc_now(),
        "duration_sec": float(time.perf_counter() - start),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
    }
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch one dynamic 2x2 seed across four GPUs with tqdm completion progress.")
    parser.add_argument("--seed", type=int, default=1601)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--output-root", default="artifacts/dynamic_2x2_convergence")
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="cuda")
    parser.add_argument("--eval-device", choices=["auto", "cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--eval-workers", type=int, default=2)
    parser.add_argument("--gpu-ids", nargs="+", default=["0", "1", "2", "3"])
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.gpu_ids) < len(MODELS):
        raise SystemExit(f"Need at least {len(MODELS)} GPU ids, got {args.gpu_ids}")
    output_root = Path(args.output_root)
    dataset_path = Path(args.dataset_path) if args.dataset_path is not None else output_root / "demos" / "dynamic_pick_place_demos.pt"
    if not dataset_path.exists():
        raise SystemExit(
            f"Missing dataset: {dataset_path}\n"
            "Generate it first with: python mujoco_dynamic_2x2_convergence.py generate-demos --episodes 500"
        )
    log_dir = output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    jobs: list[dict[str, Any]] = []
    for model, gpu in zip(MODELS, args.gpu_ids, strict=True):
        command = [
            args.python_bin,
            "mujoco_dynamic_2x2_convergence.py",
            "run-one",
            "--model",
            model,
            "--train-seed",
            str(args.seed),
            "--dataset-path",
            str(dataset_path),
            "--output-root",
            str(output_root),
            "--device",
            args.device,
            "--eval-device",
            args.eval_device,
            "--eval-workers",
            str(args.eval_workers),
        ]
        jobs.append(
            {
                "model": model,
                "seed": args.seed,
                "gpu": gpu,
                "command": command,
                "stdout_log": str(log_dir / f"{model}_seed{args.seed}.stdout.log"),
                "stderr_log": str(log_dir / f"{model}_seed{args.seed}.stderr.log"),
                "metadata_path": str(output_root / "runs" / f"{model}_seed{args.seed}" / "child_process_metadata.json"),
            }
        )
    failures: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
        futures = {executor.submit(_run_job, job): job for job in jobs}
        future_iter = tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"dynamic 2x2 seed={args.seed}",
            unit="model",
            disable=not args.progress,
        )
        for future in future_iter:
            payload = future.result()
            label = f"{payload['model']} gpu={payload['gpu']}"
            if payload["exit_code"] == 0:
                future_iter.set_postfix_str(f"done {label}")
            else:
                future_iter.set_postfix_str(f"failed {label}")
                failures.append(payload)
    if failures:
        print("At least one dynamic 2x2 run failed:", file=sys.stderr)
        for failure in failures:
            print(
                f"- {failure['model']} seed={failure['seed']} gpu={failure['gpu']} "
                f"exit={failure['exit_code']} stderr={failure['stderr_log']}",
                file=sys.stderr,
            )
            tail = _tail(Path(failure["stderr_log"]))
            if tail:
                print(tail, file=sys.stderr)
        raise SystemExit(1)
    subprocess.run(
        [args.python_bin, "mujoco_dynamic_2x2_convergence.py", "summarize", "--output-root", str(output_root)],
        check=True,
    )


if __name__ == "__main__":
    main()
