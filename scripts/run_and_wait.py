from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_and_wait(
    command: list[str],
    stdout_log: Path,
    stderr_log: Path,
    metadata_path: Path,
    stream_output: bool = False,
) -> int:
    stdout_log.parent.mkdir(parents=True, exist_ok=True)
    stderr_log.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    started_at = _utc_now()
    start = time.perf_counter()
    if stream_output:
        result = subprocess.run(command, check=False)
    else:
        with stdout_log.open("w", encoding="utf-8") as stdout_file, stderr_log.open("w", encoding="utf-8") as stderr_file:
            result = subprocess.run(command, stdout=stdout_file, stderr=stderr_file, check=False)
    finished_at = _utc_now()
    metadata = {
        "command": command,
        "exit_code": int(result.returncode),
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_sec": float(time.perf_counter() - start),
        "stdout_log": None if stream_output else str(stdout_log),
        "stderr_log": None if stream_output else str(stderr_log),
        "stream_output": stream_output,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return int(result.returncode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one subprocess, block until it exits, and record metadata.")
    parser.add_argument("--stdout-log", required=True)
    parser.add_argument("--stderr-log", required=True)
    parser.add_argument("--metadata-path", required=True)
    parser.add_argument("--stream-output", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("No command supplied after --")
    exit_code = run_and_wait(
        command=command,
        stdout_log=Path(args.stdout_log),
        stderr_log=Path(args.stderr_log),
        metadata_path=Path(args.metadata_path),
        stream_output=args.stream_output,
    )
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
