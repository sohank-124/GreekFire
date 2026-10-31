"""Fixed, non-optimizing sensitivity runs for the existing GreekFire engine.

Print commands: python robustness_matrix.py
Execute serially: python robustness_matrix.py --run [--max-steps 12]
"""

import argparse
import csv
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results" / "corrected" / "robustness"
CONFIGS = (
    ("reference", "252", "10", "equal", "0.5"),
    ("shorter_training", "126", "10", "equal", "0.5"),
    ("fewer_longs", "252", "5", "equal", "0.5"),
    ("rank_weights", "252", "10", "rank_inv_vol", "0.5"),
    ("model_blend", "252", "10", "equal", "0.75"),
)
ARTIFACTS = ("validation_holdout_A1_steps.csv", "validation_holdout_summary.csv")


def command(config, max_steps):
    _, training, positions, weighting, rf_weight = config
    result = ["julia", "--threads=1", "--project=.", "main.jl", "validate",
              "--holdout-start=2020-01-01", f"--train-window={training}",
              f"--top-n={positions}", f"--weighting-method={weighting}",
              "--transaction-cost-bps=10", "--slippage-bps=5", f"--rf-weight={rf_weight}"]
    if max_steps is not None:
        result.append(f"--max-steps={max_steps}")
    return result


def completed_export(steps_path, steps_mtime, summary_path, summary_mtime, expected_steps):
    if (not steps_path.exists() or not summary_path.exists() or
            steps_path.stat().st_mtime_ns <= steps_mtime or
            summary_path.stat().st_mtime_ns <= summary_mtime):
        return False
    try:
        with summary_path.open(newline="", encoding="utf-8") as input_file:
            row = next(csv.DictReader(input_file))
        step_count = int(row["n_steps"])
        with steps_path.open(newline="", encoding="utf-8") as input_file:
            steps = list(csv.DictReader(input_file))
    except (OSError, StopIteration, KeyError, ValueError):
        return False
    return ((step_count == expected_steps if expected_steps is not None else step_count >= 60) and
            len(steps) == step_count and steps[-1]["date"] == row["date_end"])


def self_test():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        steps = root / "steps.csv"
        summary = root / "summary.csv"
        steps.write_text("date\n2020-01-01\n", encoding="utf-8")
        summary.write_text("n_steps,date_end\n1,2020-01-01\n", encoding="utf-8")
        assert completed_export(steps, 0, summary, 0, 1)
        assert not completed_export(steps, steps.stat().st_mtime_ns, summary, 0, 1)
        assert not completed_export(steps, 0, summary, 0, 12)
        summary.write_text("n_steps,date_end\n2,2020-01-01\n", encoding="utf-8")
        assert not completed_export(steps, 0, summary, 0, 2)
        summary.write_text("n_steps\n", encoding="utf-8")
        assert not completed_export(steps, 0, summary, 0, 1)
    print("Matrix archive-completeness checks passed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Explicitly run five serial one-thread Julia experiments")
    parser.add_argument("--max-steps", type=int, help="Smoke test only; not a full robustness result")
    parser.add_argument("--force", action="store_true", help="Rerun even completed full-period archives")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps must be positive")
    reference_path = OUTPUT / "reference_validation_holdout_summary.csv"
    if args.max_steps is not None:
        expected_steps = args.max_steps
    elif reference_path.exists():
        with reference_path.open(newline="", encoding="utf-8") as input_file:
            expected_steps = int(next(csv.DictReader(input_file))["n_steps"])
    else:
        expected_steps = None
    for config in CONFIGS:
        name = config[0]
        args_list = command(config, args.max_steps)
        print(name, ":", " ".join(args_list), flush=True)
        if not args.run:
            continue
        archive_summary = OUTPUT / f"{name}_validation_holdout_summary.csv"
        if args.max_steps is None and not args.force and archive_summary.exists():
            with archive_summary.open(newline="", encoding="utf-8") as input_file:
                archived = next(csv.DictReader(input_file))
            if int(archived["n_steps"]) >= 300 and archived["date_end"] == "2026-04-08":
                print(f"{name}: reusing completed {archived['n_steps']}-step archive", flush=True)
                continue
        source = ROOT / "results" / "corrected" / ARTIFACTS[0]
        previous_mtime = source.stat().st_mtime_ns if source.exists() else 0
        summary = ROOT / "results" / "corrected" / ARTIFACTS[1]
        previous_summary_mtime = summary.stat().st_mtime_ns if summary.exists() else 0
        OUTPUT.mkdir(parents=True, exist_ok=True)
        log_path = OUTPUT / f"{name}.log"
        with log_path.open("w", encoding="utf-8") as log_file:
            with subprocess.Popen(args_list, cwd=ROOT, stdout=log_file,
                                  stderr=subprocess.STDOUT, env=os.environ.copy()) as process:
                started = time.monotonic()
                completed_at = None
                ended_after_export = False
                while True:
                    try:
                        exit_code = process.wait(timeout=20)
                        break
                    except subprocess.TimeoutExpired:
                        if completed_export(source, previous_mtime, summary,
                                            previous_summary_mtime, expected_steps):
                            completed_at = completed_at or time.monotonic()
                            if time.monotonic() - completed_at >= 120:
                                print(f"{name}: exports complete; stopping stalled Julia cleanup", flush=True)
                                if os.name == "nt":
                                    subprocess.run(["taskkill", "/T", "/F", "/PID", str(process.pid)],
                                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                                   check=True)
                                else:
                                    process.terminate()
                                exit_code = process.wait(timeout=30)
                                ended_after_export = True
                                break
                        print(f"{name}: running {int(time.monotonic() - started)}s; details in {log_path}",
                              flush=True)
                if exit_code != 0 and not ended_after_export:
                    raise subprocess.CalledProcessError(exit_code, args_list)
        if not completed_export(source, previous_mtime, summary,
                                previous_summary_mtime, expected_steps):
            raise RuntimeError(f"Incomplete or stale result for {name}; refusing to archive")
        if not source.exists() or source.stat().st_mtime_ns <= previous_mtime:
            raise RuntimeError(f"No new step export for {name}; refusing to archive stale data")
        OUTPUT.mkdir(parents=True, exist_ok=True)
        for artifact in ARTIFACTS:
            shutil.copy2(ROOT / "results" / "corrected" / artifact, OUTPUT / f"{name}_{artifact}")
        with (ROOT / "results" / "corrected" / ARTIFACTS[1]).open(newline="", encoding="utf-8") as input_file:
            result = next(csv.DictReader(input_file))
        print(f"{name}: {result['n_steps']} valid historical steps; net CAGR {result['cagr']}", flush=True)
    if not args.run:
        print("Dry run only: no training or artifacts written.")


if __name__ == "__main__":
    main()