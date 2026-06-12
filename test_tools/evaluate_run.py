from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from csv_metrics import evaluate_task, summarize_tasks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate all task predictions in a run directory.")
    parser.add_argument("run_dir", type=Path, help="Run directory under artifacts/runs, for example artifacts/runs/psrr_react_sc_v14")
    parser.add_argument(
        "--gold-root",
        type=Path,
        default=Path("data/public/output"),
        help="Gold root directory containing task_x/gold.csv",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional path for per-task metric CSV output",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path for full JSON output",
    )
    return parser


def task_dirs(run_dir: Path) -> list[Path]:
    return sorted(path for path in run_dir.iterdir() if path.is_dir() and path.name.startswith("task_"))


def write_csv(output_csv: Path, rows: list[dict[str, object]]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else [
        "task_id",
        "prediction_csv",
        "gold_csv",
        "matched_columns",
        "gold_columns",
        "predicted_columns",
        "extra_columns",
        "recall",
        "accuracy",
        "rouge",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = build_parser().parse_args()
    metrics = []
    for task_dir in task_dirs(args.run_dir):
        task_id = task_dir.name
        prediction_csv = task_dir / "prediction.csv"
        gold_csv = args.gold_root / task_id / "gold.csv"
        metrics.append(evaluate_task(task_id, prediction_csv, gold_csv))

    per_task_rows = [item.to_flat_dict() for item in metrics]
    summary = summarize_tasks(metrics)
    payload = {
        "run_dir": str(args.run_dir),
        "gold_root": str(args.gold_root),
        "summary": summary,
        "tasks": per_task_rows,
    }

    if args.output_csv is not None:
        write_csv(args.output_csv, per_task_rows)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()