from __future__ import annotations

import argparse
import json
from pathlib import Path

from csv_metrics import evaluate_task


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a single prediction.csv against gold.csv.")
    parser.add_argument("--task-id", required=True, help="Task identifier, for example task_11")
    parser.add_argument("--prediction-csv", required=True, type=Path, help="Path to prediction.csv")
    parser.add_argument("--gold-csv", required=True, type=Path, help="Path to gold.csv")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metrics = evaluate_task(args.task_id, args.prediction_csv, args.gold_csv)
    print(json.dumps(metrics.to_flat_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()