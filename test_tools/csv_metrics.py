from __future__ import annotations

import csv
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ColumnMetrics:
    matched_columns: int
    gold_columns: int
    predicted_columns: int
    extra_columns: int
    recall: float
    accuracy: float


@dataclass(slots=True)
class TextMetrics:
    rouge: float


@dataclass(slots=True)
class TaskMetrics:
    task_id: str
    prediction_csv: str | None
    gold_csv: str | None
    column_metrics: ColumnMetrics
    text_metrics: TextMetrics

    def to_flat_dict(self) -> dict[str, Any]:
        payload = {
            "task_id": self.task_id,
            "prediction_csv": self.prediction_csv,
            "gold_csv": self.gold_csv,
        }
        payload.update(asdict(self.column_metrics))
        payload.update(asdict(self.text_metrics))
        return payload


def read_csv_rows(csv_path: Path) -> list[list[str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return [[cell.strip() for cell in row] for row in csv.reader(handle)]


def split_header_and_rows(table: list[list[str]]) -> tuple[list[str], list[list[str]]]:
    if not table:
        return [], []
    return table[0], table[1:]


def pad_rows(rows: list[list[str]], width: int) -> list[list[str]]:
    return [row + [""] * (width - len(row)) for row in rows]


def table_width(header: list[str], rows: list[list[str]]) -> int:
    row_width = max((len(row) for row in rows), default=0)
    return max(len(header), row_width)


def column_signatures(header: list[str], rows: list[list[str]]) -> list[tuple[str, ...]]:
    width = table_width(header, rows)
    if width == 0:
        return []
    padded_rows = pad_rows(rows, width)
    signatures: list[tuple[str, ...]] = []
    for index in range(width):
        signatures.append(tuple(sorted(row[index] for row in padded_rows)))
    return signatures


def safe_ratio(numerator: int, denominator: int, *, empty_value: float) -> float:
    if denominator == 0:
        return empty_value
    return numerator / denominator


def compute_column_metrics(prediction_table: list[list[str]], gold_table: list[list[str]]) -> ColumnMetrics:
    prediction_header, prediction_rows = split_header_and_rows(prediction_table)
    gold_header, gold_rows = split_header_and_rows(gold_table)

    prediction_signatures = Counter(column_signatures(prediction_header, prediction_rows))
    gold_signatures = Counter(column_signatures(gold_header, gold_rows))
    matched_columns = sum(min(prediction_signatures[key], gold_signatures[key]) for key in gold_signatures)
    predicted_columns = table_width(prediction_header, prediction_rows)
    gold_columns = table_width(gold_header, gold_rows)
    extra_columns = max(predicted_columns - matched_columns, 0)

    empty_value = 1.0 if predicted_columns == 0 and gold_columns == 0 else 0.0
    return ColumnMetrics(
        matched_columns=matched_columns,
        gold_columns=gold_columns,
        predicted_columns=predicted_columns,
        extra_columns=extra_columns,
        recall=safe_ratio(matched_columns, gold_columns, empty_value=empty_value),
        accuracy=safe_ratio(matched_columns, predicted_columns, empty_value=empty_value),
    )


def flatten_cells(table: list[list[str]]) -> list[str]:
    _, rows = split_header_and_rows(table)
    return sorted(cell for row in rows for cell in row if cell != "")


def ngrams(tokens: list[str], order: int) -> Counter[tuple[str, ...]]:
    if len(tokens) < order:
        return Counter()
    return Counter(tuple(tokens[index : index + order]) for index in range(len(tokens) - order + 1))


def rouge_n_f1(prediction_tokens: list[str], gold_tokens: list[str], order: int) -> float:
    if len(prediction_tokens) < order and len(gold_tokens) < order:
        return 1.0 if prediction_tokens == gold_tokens else 0.0
    if len(prediction_tokens) < order or len(gold_tokens) < order:
        return 0.0
    prediction_ngrams = ngrams(prediction_tokens, order)
    gold_ngrams = ngrams(gold_tokens, order)
    overlap = sum(min(prediction_ngrams[key], gold_ngrams[key]) for key in gold_ngrams)
    prediction_total = sum(prediction_ngrams.values())
    gold_total = sum(gold_ngrams.values())
    precision = overlap / prediction_total if prediction_total else 0.0
    recall = overlap / gold_total if gold_total else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def lcs_length(left: list[str], right: list[str]) -> int:
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def rouge_l_f1(prediction_tokens: list[str], gold_tokens: list[str]) -> float:
    if not prediction_tokens and not gold_tokens:
        return 1.0
    if not prediction_tokens or not gold_tokens:
        return 0.0
    lcs = lcs_length(prediction_tokens, gold_tokens)
    precision = lcs / len(prediction_tokens)
    recall = lcs / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def bleu_score(prediction_tokens: list[str], gold_tokens: list[str], max_order: int = 4) -> float:
    if not prediction_tokens and not gold_tokens:
        return 1.0
    if not prediction_tokens or not gold_tokens:
        return 0.0

    effective_max_order = min(max_order, len(prediction_tokens), len(gold_tokens))
    if effective_max_order == 0:
        return 0.0

    precisions: list[float] = []
    for order in range(1, effective_max_order + 1):
        prediction_ngrams = ngrams(prediction_tokens, order)
        gold_ngrams = ngrams(gold_tokens, order)
        total = sum(prediction_ngrams.values())
        if total == 0:
            precisions.append(0.0)
            continue
        overlap = sum(min(count, gold_ngrams[gram]) for gram, count in prediction_ngrams.items())
        precisions.append((overlap + 1.0) / (total + 1.0))

    if min(precisions) == 0.0:
        return 0.0

    prediction_length = len(prediction_tokens)
    gold_length = len(gold_tokens)
    if prediction_length > gold_length:
        brevity_penalty = 1.0
    else:
        brevity_penalty = math.exp(1 - gold_length / prediction_length)

    geometric_mean = math.exp(sum(math.log(value) for value in precisions) / effective_max_order)
    return brevity_penalty * geometric_mean


def compute_text_metrics(prediction_table: list[list[str]], gold_table: list[list[str]]) -> TextMetrics:
    prediction_tokens = flatten_cells(prediction_table)
    gold_tokens = flatten_cells(gold_table)
    return TextMetrics(
        rouge=rouge_n_f1(prediction_tokens, gold_tokens, order=1),
    )


def evaluate_task(task_id: str, prediction_csv: Path | None, gold_csv: Path | None) -> TaskMetrics:
    prediction_table = read_csv_rows(prediction_csv) if prediction_csv and prediction_csv.exists() else []
    gold_table = read_csv_rows(gold_csv) if gold_csv and gold_csv.exists() else []
    return TaskMetrics(
        task_id=task_id,
        prediction_csv=str(prediction_csv) if prediction_csv else None,
        gold_csv=str(gold_csv) if gold_csv else None,
        column_metrics=compute_column_metrics(prediction_table, gold_table),
        text_metrics=compute_text_metrics(prediction_table, gold_table),
    )


def summarize_tasks(task_metrics: list[TaskMetrics]) -> dict[str, float | int]:
    if not task_metrics:
        return {
            "task_count": 0,
            "matched_columns": 0,
            "gold_columns": 0,
            "predicted_columns": 0,
            "extra_columns": 0,
            "recall": 0.0,
            "accuracy": 0.0,
            "rouge": 0.0,
        }

    matched_columns = sum(item.column_metrics.matched_columns for item in task_metrics)
    gold_columns = sum(item.column_metrics.gold_columns for item in task_metrics)
    predicted_columns = sum(item.column_metrics.predicted_columns for item in task_metrics)
    extra_columns = sum(item.column_metrics.extra_columns for item in task_metrics)

    return {
        "task_count": len(task_metrics),
        "matched_columns": matched_columns,
        "gold_columns": gold_columns,
        "predicted_columns": predicted_columns,
        "extra_columns": extra_columns,
        "recall": sum(item.column_metrics.recall for item in task_metrics) / len(task_metrics),
        "accuracy": sum(item.column_metrics.accuracy for item in task_metrics) / len(task_metrics),
        "rouge": sum(item.text_metrics.rouge for item in task_metrics) / len(task_metrics),
    }