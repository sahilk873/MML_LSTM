import argparse
import json
import os
import shlex
import subprocess
import sys
from typing import Dict

from polypharmacy import utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate on refined ground-truth CSVs and compare against baseline."
    )
    parser.add_argument("--baseline-output-dir", default="artifacts_baseline")
    parser.add_argument("--refined-output-dir", default="artifacts_refined")
    parser.add_argument("--baseline-indications", default="indications_norm_dedup.csv")
    parser.add_argument(
        "--baseline-contraindications", default="contraindications_norm_dedup.csv"
    )
    parser.add_argument(
        "--refined-indications",
        default="artifacts/refined_gt/refined_indications.csv",
    )
    parser.add_argument(
        "--refined-contraindications",
        default="contraindications_norm_dedup.csv",
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--kg", default="kg_edges.parquet")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    return parser.parse_args()


def _run_command(cmd: str) -> None:
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def _extract_metrics(eval_stdout: str) -> Dict[str, float]:
    lines = [line.strip() for line in eval_stdout.splitlines() if line.strip()]
    test_lines = [line for line in lines if line.startswith("Test")]
    if not test_lines:
        return {}
    latest = test_lines[-1]
    metrics: Dict[str, float] = {}
    for token in latest.split("|"):
        token = token.strip()
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        key = key.strip()
        value = value.strip()
        try:
            metrics[key] = float(value)
        except ValueError:
            continue
    return metrics


def _run_eval_capture(cmd: str) -> Dict[str, float]:
    print(f"Running: {cmd}")
    proc = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
    print(proc.stdout)
    return _extract_metrics(proc.stdout)


def _build_optional_flag(name: str, value: object) -> str:
    if value is None:
        return ""
    return f" --{name} {shlex.quote(str(value))}"


def main() -> None:
    args = parse_args()

    py = shlex.quote(sys.executable)
    baseline_train_cmd = (
        f"{py} train.py"
        f" --indications {shlex.quote(args.baseline_indications)}"
        f" --contraindications {shlex.quote(args.baseline_contraindications)}"
        f" --kg {shlex.quote(args.kg)}"
        f" --output-dir {shlex.quote(args.baseline_output_dir)}"
        f"{_build_optional_flag('config', args.config)}"
        f"{_build_optional_flag('batch-size', args.batch_size)}"
        f"{_build_optional_flag('epochs', args.epochs)}"
    )
    _run_command(baseline_train_cmd)

    train_cmd = (
        f"{py} train.py"
        f" --indications {shlex.quote(args.refined_indications)}"
        f" --contraindications {shlex.quote(args.refined_contraindications)}"
        f" --kg {shlex.quote(args.kg)}"
        f" --output-dir {shlex.quote(args.refined_output_dir)}"
        f"{_build_optional_flag('config', args.config)}"
        f"{_build_optional_flag('batch-size', args.batch_size)}"
        f"{_build_optional_flag('epochs', args.epochs)}"
    )
    _run_command(train_cmd)

    refined_eval_cmd = (
        f"{py} evaluate.py"
        f" --indications {shlex.quote(args.refined_indications)}"
        f" --contraindications {shlex.quote(args.refined_contraindications)}"
        f" --kg {shlex.quote(args.kg)}"
        f" --output-dir {shlex.quote(args.refined_output_dir)}"
        f"{_build_optional_flag('config', args.config)}"
    )
    refined_metrics = _run_eval_capture(refined_eval_cmd)

    baseline_eval_cmd = (
        f"{py} evaluate.py"
        f" --indications {shlex.quote(args.baseline_indications)}"
        f" --contraindications {shlex.quote(args.baseline_contraindications)}"
        f" --output-dir {shlex.quote(args.baseline_output_dir)}"
        f" --kg {shlex.quote(args.kg)}"
        f"{_build_optional_flag('config', args.config)}"
    )
    baseline_metrics = _run_eval_capture(baseline_eval_cmd)

    comparison = {
        "baseline_indications": args.baseline_indications,
        "baseline_contraindications": args.baseline_contraindications,
        "refined_indications": args.refined_indications,
        "refined_contraindications": args.refined_contraindications,
        "baseline_output_dir": args.baseline_output_dir,
        "refined_output_dir": args.refined_output_dir,
        "baseline_test_metrics": baseline_metrics,
        "refined_test_metrics": refined_metrics,
    }

    report_dir = "artifacts/refined_gt"
    utils.ensure_dir(report_dir)
    json_path = os.path.join(report_dir, "experiment_comparison.json")
    md_path = os.path.join(report_dir, "experiment_comparison.md")

    utils.save_json(json_path, comparison)
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("# Baseline vs Refined Comparison\n\n")
        handle.write(f"- Baseline artifacts: `{args.baseline_output_dir}`\n")
        handle.write(f"- Refined artifacts: `{args.refined_output_dir}`\n\n")
        handle.write("## Baseline Test Metrics\n")
        handle.write("```json\n")
        handle.write(json.dumps(baseline_metrics, indent=2, sort_keys=True))
        handle.write("\n```\n\n")
        handle.write("## Refined Test Metrics\n")
        handle.write("```json\n")
        handle.write(json.dumps(refined_metrics, indent=2, sort_keys=True))
        handle.write("\n```\n")

    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
