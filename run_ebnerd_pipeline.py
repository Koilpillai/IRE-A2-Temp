"""EB-NeRD automated pipeline runner.

1. Safely waits for PID 22924 (feature builder) to exit and verifies
   feature_store/ebnerd/behavioral_features_val.parquet footer.
2. Runs all EB-NeRD pipeline stages strictly for '--dataset ebnerd', avoiding
   any redundant re-runs of MIND.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq

REPO_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable
VAL_FILE = REPO_DIR / "feature_store" / "ebnerd" / "behavioral_features_val.parquet"
FEATURE_BUILDER_PID = 22924


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def is_pid_running(pid: int) -> bool:
    try:
        import psutil
        return psutil.pid_exists(pid)
    except Exception:
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}"],
            capture_output=True, text=True, check=False
        )
        return str(pid) in out.stdout


def wait_for_features() -> None:
    log(f"Waiting for PID {FEATURE_BUILDER_PID} to exit and EB-NeRD val features to finalize...")
    while True:
        alive = is_pid_running(FEATURE_BUILDER_PID)
        if not alive:
            try:
                pf = pq.ParquetFile(VAL_FILE)
                log(f"EB-NeRD val features verified! ({pf.metadata.num_rows:,} rows across {pf.metadata.num_row_groups} row groups)")
                return
            except Exception as e:
                log(f"PID {FEATURE_BUILDER_PID} exited but Parquet file not ready yet ({e}). Retrying in 10s...")
        time.sleep(10)


def run_stage(module: str, args: list[str]) -> None:
    cmd = [PYTHON, "-u", "-m", module] + args
    log(f"=== Running: {' '.join(cmd)} ===")
    t0 = time.time()
    ret = subprocess.run(cmd, cwd=REPO_DIR)
    elapsed = time.time() - t0
    if ret.returncode != 0:
        log(f"ERROR: Stage {module} failed with exit code {ret.returncode} after {elapsed:.1f}s!")
        sys.exit(ret.returncode)
    log(f"Completed {module} successfully in {elapsed:.1f}s.")


def main() -> None:
    wait_for_features()

    stages = [
        ("pipeline.rerank", ["--dataset", "ebnerd"]),
        ("pipeline.ablation", ["--dataset", "ebnerd"]),
        ("pipeline.anti_gaming", ["--dataset", "ebnerd"]),
        ("pipeline.evaluate", ["--dataset", "ebnerd"]),
        ("pipeline.serving_scale", ["--dataset", "ebnerd"]),
        ("pipeline.generate_submission", ["--dataset", "ebnerd"]),
        ("pytest", ["pipeline/tests/", "-v"]),
    ]

    for module, args in stages:
        run_stage(module, args)

    log("=== ALL EB-NERD PIPELINE STAGES COMPLETED SUCCESSFULLY ===")


if __name__ == "__main__":
    main()
