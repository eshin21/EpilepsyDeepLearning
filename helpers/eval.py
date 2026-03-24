from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .data_io import ensure_dir, write_json


def select_best_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if len(thresholds) == 0:
        return 0.5
    f1 = (2 * precision[:-1] * recall[:-1]) / np.maximum(precision[:-1] + recall[:-1], 1e-8)
    best_index = int(np.nanargmax(np.nan_to_num(f1)))
    return float(thresholds[best_index])


def compute_binary_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / max(tn + fp, 1)
    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else float("nan"),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "n_rows": int(len(y_true)),
    }
    return metrics


def build_prediction_frame(
    prediction_df: pd.DataFrame,
    threshold: float,
    protocol: str,
    outer_fold_id: int,
    train_mode: str,
    test_mode: str,
    checkpoint_path: Path,
) -> pd.DataFrame:
    frame = prediction_df.copy()
    frame["threshold"] = float(threshold)
    frame["y_pred"] = (frame["y_score"] >= threshold).astype(int)
    frame["protocol"] = protocol
    frame["outer_fold_id"] = int(outer_fold_id)
    frame["train_mode"] = train_mode
    frame["test_mode"] = test_mode
    frame["checkpoint_path"] = str(checkpoint_path)
    return frame[
        ["row_id", "protocol", "outer_fold_id", "train_mode", "test_mode", "y_true", "y_score", "y_pred", "threshold", "checkpoint_path"]
    ].sort_values("row_id")


def save_evaluation_bundle(output_dir: Path, prediction_frame: pd.DataFrame, metrics: dict[str, float]) -> None:
    ensure_dir(output_dir)
    predictions_path = output_dir / "predictions.parquet"
    metrics_path = output_dir / "metrics.json"
    confusion_path = output_dir / "confusion_matrix.csv"
    roc_path = output_dir / "roc_curve.csv"
    pr_path = output_dir / "pr_curve.csv"
    threshold_path = output_dir / "threshold.json"

    prediction_frame.to_parquet(predictions_path, index=False)
    write_json(metrics_path, metrics)
    write_json(threshold_path, {"threshold": metrics["threshold"]})

    confusion_df = pd.DataFrame(
        [
            {"actual": 0, "predicted": 0, "count": metrics["tn"]},
            {"actual": 0, "predicted": 1, "count": metrics["fp"]},
            {"actual": 1, "predicted": 0, "count": metrics["fn"]},
            {"actual": 1, "predicted": 1, "count": metrics["tp"]},
        ]
    )
    confusion_df.to_csv(confusion_path, index=False)

    fpr, tpr, roc_thresholds = roc_curve(prediction_frame["y_true"], prediction_frame["y_score"])
    pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": roc_thresholds}).to_csv(roc_path, index=False)

    precision, recall, pr_thresholds = precision_recall_curve(prediction_frame["y_true"], prediction_frame["y_score"])
    pr_df = pd.DataFrame(
        {
            "precision": precision,
            "recall": recall,
            "threshold": np.append(pr_thresholds, np.nan),
        }
    )
    pr_df.to_csv(pr_path, index=False)
