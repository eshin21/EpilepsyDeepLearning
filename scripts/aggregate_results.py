#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib
import pandas as pd
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from helpers.data_io import FOLD7_DIR, ensure_dir


METRIC_COLUMNS = ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "n_rows"]


def collect_metric_rows(project_root: Path = ROOT) -> pd.DataFrame:
    evaluation_root = project_root / "fold6_evaluation"
    rows: list[dict[str, object]] = []
    for metrics_path in sorted(evaluation_root.glob("*/*/outer_fold_*/*/metrics.json")):
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        test_mode = metrics_path.parent.name
        outer_fold = metrics_path.parent.parent.name
        train_mode = metrics_path.parent.parent.parent.name
        protocol = metrics_path.parent.parent.parent.parent.name
        row = {
            "protocol": protocol,
            "train_mode": train_mode,
            "test_mode": test_mode,
            "outer_fold": outer_fold,
            "metrics_path": str(metrics_path),
        }
        for key in METRIC_COLUMNS + ["threshold", "tn", "fp", "fn", "tp"]:
            row[key] = payload.get(key)
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["protocol", "train_mode", "test_mode", "outer_fold"] + METRIC_COLUMNS)
    return pd.DataFrame(rows)


def aggregate_results(project_root: Path = ROOT) -> pd.DataFrame:
    ensure_dir(FOLD7_DIR)
    ensure_dir(FOLD7_DIR / "summary_figures")
    metric_rows = collect_metric_rows(project_root)
    metric_rows.to_csv(FOLD7_DIR / "per_fold_metrics.csv", index=False)
    if metric_rows.empty:
        (FOLD7_DIR / "applications_discussion.md").write_text(
            "# Applications Discussion\n\nNo evaluation outputs are available yet.\n",
            encoding="utf-8",
        )
        return metric_rows

    grouped = (
        metric_rows.groupby(["protocol", "train_mode", "test_mode"])
        .agg(
            n_folds=("outer_fold", "nunique"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            precision_mean=("precision", "mean"),
            precision_std=("precision", "std"),
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
            specificity_mean=("specificity", "mean"),
            specificity_std=("specificity", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            n_rows_mean=("n_rows", "mean"),
        )
        .reset_index()
        .sort_values(["protocol", "train_mode", "test_mode"])
    )
    grouped.to_csv(FOLD7_DIR / "main_results.csv", index=False)
    grouped.to_latex(FOLD7_DIR / "main_table.tex", index=False, float_format=lambda value: f"{value:.4f}")
    _plot_metric(grouped, "f1_mean", "Mean F1 by protocol", FOLD7_DIR / "summary_figures" / "f1_by_protocol.png")
    _plot_metric(grouped, "roc_auc_mean", "Mean ROC-AUC by protocol", FOLD7_DIR / "summary_figures" / "auc_by_protocol.png")
    _plot_confusion_overview(
        grouped,
        metric_rows,
        test_mode="balanced_50_50",
        output_path=FOLD7_DIR / "summary_figures" / "confusion_matrices_balanced.png",
    )
    _plot_confusion_overview(
        grouped,
        metric_rows,
        test_mode="unbalanced_20_80",
        output_path=FOLD7_DIR / "summary_figures" / "confusion_matrices_unbalanced.png",
    )
    _write_discussion(grouped, FOLD7_DIR / "applications_discussion.md")
    _write_report_notes(grouped, FOLD7_DIR / "report_ready_notes.md")
    return grouped


def _plot_metric(grouped: pd.DataFrame, metric: str, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.8, 5.4))
    labels = []
    values = []
    colors = []
    color_map = {
        "balanced_50_50|balanced_50_50": "#4C78A8",
        "balanced_50_50|unbalanced_20_80": "#72B7B2",
        "unbalanced_20_80|balanced_50_50": "#F58518",
        "unbalanced_20_80|unbalanced_20_80": "#E45756",
    }
    for _, row in grouped.iterrows():
        combo = f"{row['train_mode']}|{row['test_mode']}"
        labels.append(f"{row['protocol']}\n{row['train_mode']}\n{row['test_mode']}")
        values.append(float(row[metric]))
        colors.append(color_map.get(combo, "#7f7f7f"))
    positions = list(range(len(labels)))
    ax.bar(positions, values, color=colors, edgecolor="#384860", linewidth=0.5, alpha=0.92)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=42, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Mean F1" if metric == "f1_mean" else "Mean ROC-AUC")
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color="#1f2937")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#6b7280")
    ax.spines["bottom"].set_color("#6b7280")
    ax.tick_params(colors="#374151")
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.28, color="#9aa5b1")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, facecolor="white")
    plt.close(fig)


def _plot_confusion_overview(
    grouped: pd.DataFrame,
    metric_rows: pd.DataFrame,
    test_mode: str,
    output_path: Path,
) -> None:
    protocols = ["window", "seizure", "patient"]
    best_rows: dict[str, pd.Series] = {}
    for protocol in protocols:
        subset = grouped[(grouped["protocol"] == protocol) & (grouped["test_mode"] == test_mode)].copy()
        if subset.empty:
            continue
        subset = subset.sort_values(["f1_mean", "roc_auc_mean"], ascending=False)
        best_rows[protocol] = subset.iloc[0]

    if not best_rows:
        return

    fig, axes = plt.subplots(1, len(protocols), figsize=(16.8, 5.4), gridspec_kw={"wspace": 0.42})
    if len(protocols) == 1:
        axes = [axes]
    fig.subplots_adjust(left=0.06, right=0.88, bottom=0.16, top=0.78, wspace=0.42)

    images = []
    for ax, protocol in zip(axes, protocols):
        row = best_rows.get(protocol)
        if row is None:
            ax.axis("off")
            continue

        selected = metric_rows[
            (metric_rows["protocol"] == protocol)
            & (metric_rows["train_mode"] == row["train_mode"])
            & (metric_rows["test_mode"] == test_mode)
        ]
        cm = np.array(
            [
                [float(selected["tn"].fillna(0).sum()), float(selected["fp"].fillna(0).sum())],
                [float(selected["fn"].fillna(0).sum()), float(selected["tp"].fillna(0).sum())],
            ],
            dtype=float,
        )
        row_sums = cm.sum(axis=1, keepdims=True)
        normalized = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)

        image = ax.imshow(normalized, cmap="YlGnBu", vmin=0.0, vmax=1.0)
        images.append(image)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Pred 0", "Pred 1"], fontsize=10)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Actual 0", "Actual 1"], fontsize=10)
        ax.set_title(
            f"{protocol.capitalize()}\nbest train: {_short_mode(str(row['train_mode']))}",
            fontsize=10.5,
            color="#1f2937",
            fontweight="bold",
            pad=14,
        )

        for i in range(2):
            for j in range(2):
                label = f"{int(cm[i, j]):,}\n{normalized[i, j] * 100:.1f}%"
                color = "white" if normalized[i, j] >= 0.58 else "#1f2937"
                ax.text(
                    j,
                    i,
                    label,
                    ha="center",
                    va="center",
                    fontsize=10,
                    color=color,
                    fontweight="bold",
                    linespacing=1.15,
                )

        for spine in ax.spines.values():
            spine.set_color("#94a3b8")
            spine.set_linewidth(0.8)

    if images:
        cax = fig.add_axes([0.90, 0.18, 0.014, 0.62])
        cbar = fig.colorbar(images[-1], cax=cax)
        cbar.set_label("Row-normalized proportion", color="#374151")
        cbar.outline.set_edgecolor("#94a3b8")
        cbar.ax.tick_params(colors="#374151")

    title_mode = "Balanced 50/50" if test_mode == "balanced_50_50" else "Unbalanced 20/80"
    fig.suptitle(
        f"Aggregated confusion matrices by protocol ({title_mode} test)",
        fontsize=14,
        fontweight="bold",
        color="#1f2937",
        x=0.06,
        ha="left",
        y=0.94,
    )
    fig.text(
        0.06,
        0.05,
        "Rows are actual classes and columns are predicted classes. Each panel aggregates all outer folds from the best train-mode setting for that protocol and test regime.",
        fontsize=10,
        color="#475569",
    )
    fig.savefig(output_path, dpi=220, facecolor="white")
    plt.close(fig)


def _short_mode(mode: str) -> str:
    return {"balanced_50_50": "50/50", "unbalanced_20_80": "20/80"}.get(mode, mode)


def _write_discussion(grouped: pd.DataFrame, output_path: Path) -> None:
    lines = [
        "# Applications Discussion",
        "",
        "## Protocol meanings",
        "",
        "- `window`: optimistic upper bound because train/test share patient and event context.",
        "- `seizure`: personalized unseen-event evaluation for known subjects.",
        "- `patient`: population evaluation on unseen subjects.",
        "",
        "## Balanced vs unbalanced",
        "",
        "- Balanced test sets are easier to compare visually across folds.",
        "- Unbalanced test sets better expose false positives and trustworthiness under realistic class ratios.",
        "",
        "## Current summary",
        "",
    ]
    for _, row in grouped.iterrows():
        lines.append(
            f"- `{row['protocol']}` with `{row['train_mode']}` train and `{row['test_mode']}` test: "
            f"F1 `{row['f1_mean']:.4f} ± {row['f1_std']:.4f}`, "
            f"AUC `{row['roc_auc_mean']:.4f} ± {row['roc_auc_std']:.4f}`."
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report_notes(grouped: pd.DataFrame, output_path: Path) -> None:
    best_rows = grouped.sort_values(["f1_mean", "roc_auc_mean"], ascending=False).head(3)
    lines = [
        "# Report-ready Notes",
        "",
        "## Suggested narrative",
        "",
        "- Start by framing window-level validation as the upper bound.",
        "- Then compare seizure holdout as the personalized setting.",
        "- Finish with patient holdout as the most realistic population-level setting.",
        "",
        "## Top metric rows",
        "",
    ]
    for _, row in best_rows.iterrows():
        lines.append(
            f"- `{row['protocol']}` / `{row['train_mode']}` / `{row['test_mode']}` "
            f"achieved mean F1 `{row['f1_mean']:.4f}` and mean AUC `{row['roc_auc_mean']:.4f}`."
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    aggregate_results(ROOT)


if __name__ == "__main__":
    main()
