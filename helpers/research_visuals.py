from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/hhome/ricse03/Deep_Learning_Group 3/homework_fixed/.mplconfig")

import matplotlib
import numpy as np
import pandas as pd
import torch
from matplotlib import colors
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

matplotlib.use("Agg")

from .data_io import CACHE_DIR, FOLD7_DIR, PROJECT_ROOT, ensure_dir, load_master_index, load_single_window, write_json
from .model import load_trained_model


CHANNEL_NAMES = [
    "FP1-F7",
    "F7-T7",
    "T7-P7",
    "P7-O1",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
    "FP2-F8",
    "F8-T8",
    "T8-P8",
    "P8-O2",
    "FZ-CZ",
    "CZ-PZ",
    "T7-FT9",
    "FT9-FT10",
    "FT10-T8",
]

REFERENCE_PROTOCOL = "patient"
REFERENCE_TRAIN_MODE = "balanced_50_50"
REFERENCE_TEST_MODE = "balanced_50_50"
REFERENCE_OUTER_FOLD = 0


def set_research_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Serif",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
        }
    )


def reference_prediction_path(project_root: Path = PROJECT_ROOT) -> Path:
    return (
        project_root
        / "fold6_evaluation"
        / REFERENCE_PROTOCOL
        / REFERENCE_TRAIN_MODE
        / f"outer_fold_{REFERENCE_OUTER_FOLD:03d}"
        / REFERENCE_TEST_MODE
        / "predictions.parquet"
    )


def research_visuals_dir(project_root: Path = PROJECT_ROOT) -> Path:
    return ensure_dir(project_root / "fold7_results_and_reporting" / "research_visuals")


def load_window_from_row(master_index: pd.DataFrame, row_id: int) -> tuple[np.ndarray, pd.Series]:
    row = master_index.loc[master_index["row_id"] == row_id].iloc[0]
    cache_path = CACHE_DIR / f"{row['patient_id']}_windows.float32.npy"
    if cache_path.exists():
        cache = np.load(cache_path, mmap_mode="r")
        window = np.asarray(cache[int(row["window_idx_in_patient"])], dtype=np.float32)
    else:
        window = load_single_window(Path(row["source_npz_path"]), int(row["window_idx_in_patient"]))
    return window, row


def _representative_case(correct_rows: pd.DataFrame) -> pd.Series:
    target = float(correct_rows["y_score"].median())
    ranked = correct_rows.assign(_distance=(correct_rows["y_score"] - target).abs()).sort_values(["_distance", "row_id"])
    return ranked.iloc[0]


def select_reference_cases(master_index: pd.DataFrame, prediction_path: Path) -> tuple[pd.Series, pd.Series, Path]:
    prediction_df = pd.read_parquet(prediction_path)
    merged = prediction_df.merge(
        master_index[["row_id", "patient_id", "class_label", "filename", "global_interval"]],
        on="row_id",
        how="left",
    )
    tp = merged.loc[(merged["y_true"] == 1) & (merged["y_pred"] == 1)].copy()
    tn = merged.loc[(merged["y_true"] == 0) & (merged["y_pred"] == 0)].copy()
    if tp.empty or tn.empty:
        raise RuntimeError("Could not find both a true-positive and a true-negative representative case.")
    seizure_case = _representative_case(tp)
    normal_case = _representative_case(tn)
    checkpoint_path = Path(str(prediction_df["checkpoint_path"].iloc[0]))
    return seizure_case, normal_case, checkpoint_path


def _signal_norm(*windows: np.ndarray) -> colors.TwoSlopeNorm:
    merged = np.concatenate([window.reshape(-1) for window in windows])
    vmax = float(np.percentile(np.abs(merged), 99))
    vmax = max(vmax, 1.0)
    return colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)


def save_input_heatmap_pair(
    normal_window: np.ndarray,
    seizure_window: np.ndarray,
    normal_case: pd.Series,
    seizure_case: pd.Series,
    output_path: Path,
) -> None:
    set_research_style()
    ensure_dir(output_path.parent)
    norm = _signal_norm(normal_window, seizure_window)
    cmap = "RdBu_r"

    fig, axes = plt.subplots(1, 2, figsize=(18.5, 9.8), sharey=True)
    fig.subplots_adjust(left=0.14, right=0.92, bottom=0.19, top=0.84, wspace=0.22)
    for idx, (ax, window, case, title) in enumerate([
        (axes[0], normal_window, normal_case, "Non-seizure window"),
        (axes[1], seizure_window, seizure_case, "Seizure window"),
    ]):
        im = ax.imshow(window, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_title(title, fontsize=13, pad=12)
        ax.text(
            0.5,
            -0.15,
            f"row_id={int(case['row_id'])} | patient={case['patient_id']} | score={case['y_score']:.3f}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8.8,
            color="#444444",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#d9d9d9", "linewidth": 0.8, "alpha": 0.98},
        )
        ax.set_xlabel("Time index (128 samples)")
        ax.set_yticks(np.arange(len(CHANNEL_NAMES)))
        if idx == 0:
            ax.set_ylabel("EEG channel")
            ax.set_yticklabels(CHANNEL_NAMES, fontsize=8)
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", left=False, labelleft=False)
    cbar = fig.colorbar(im, ax=axes, shrink=0.88, pad=0.012)
    cbar.set_label("Amplitude (a.u.)")
    fig.suptitle("Representative single-window EEG inputs for the channel-fusion CNN baseline", fontsize=16, y=0.95)
    fig.savefig(output_path, dpi=240, facecolor="white")
    plt.close(fig)


def save_channel_fusion_architecture(output_path: Path) -> None:
    set_research_style()
    ensure_dir(output_path.parent)

    fig, ax = plt.subplots(figsize=(17, 6.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    boxes = [
        (0.02, 0.26, 0.16, 0.44, "#d8e9ff", "EEG window\n21 channels x 128 samples"),
        (0.23, 0.26, 0.18, 0.44, "#ffe3d3", "Conv1d block 1\n21 -> 16 filters\nFirst fusion stage"),
        (0.47, 0.26, 0.14, 0.44, "#fff0c9", "Conv1d block 2\n16 -> 32 filters"),
        (0.66, 0.26, 0.14, 0.44, "#e5f5de", "Conv1d block 3\n32 -> 64 filters"),
        (0.85, 0.26, 0.12, 0.44, "#efe4ff", "Pooling + linear\nBinary score"),
    ]

    for x, y, w, h, face, text in boxes:
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02", linewidth=1.3, edgecolor="#2f2f2f", facecolor=face)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=11)

    for start, end in [(0.18, 0.23), (0.41, 0.47), (0.61, 0.66), (0.80, 0.85)]:
        ax.add_patch(FancyArrowPatch((start, 0.50), (end, 0.50), arrowstyle="-|>", mutation_scale=16, linewidth=1.4, color="#4c4c4c"))

    ax.text(0.5, 0.83, "Each first-layer kernel mixes information from all 21 EEG channels.", ha="center", va="center", fontsize=12, color="#8f2d1e")
    ax.text(
        0.5,
        0.16,
        "Input-level channel fusion: multichannel EEG is combined before high-level features are formed.",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#444444",
    )
    ax.set_title("Channel-fusion CNN baseline", fontsize=16, pad=14)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def save_first_layer_weight_heatmap(checkpoint_path: Path, output_path: Path) -> None:
    set_research_style()
    ensure_dir(output_path.parent)

    model, _, _, _ = load_trained_model(checkpoint_path, device="cpu")
    first_conv = model.encoder.features[0]
    weights = first_conv.weight.detach().cpu().numpy()
    fusion_strength = np.mean(np.abs(weights), axis=2)
    channel_mean = fusion_strength.mean(axis=0)

    fig = plt.figure(figsize=(18, 9.5))
    gs = GridSpec(1, 2, width_ratios=[4.2, 1.5], wspace=0.22, figure=fig)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    im = ax0.imshow(fusion_strength, aspect="auto", cmap="magma", interpolation="nearest")
    ax0.set_title("First-layer channel-fusion strength\nmean(|weight|) over kernel time axis")
    ax0.set_xlabel("Input EEG channel")
    ax0.set_ylabel("Conv1d filter")
    ax0.set_xticks(np.arange(len(CHANNEL_NAMES)))
    ax0.set_xticklabels(CHANNEL_NAMES, rotation=90, ha="center", fontsize=8)
    ax0.set_yticks(np.arange(fusion_strength.shape[0]))
    ax0.set_yticklabels([f"F{i+1}" for i in range(fusion_strength.shape[0])], fontsize=8.5)
    cbar = fig.colorbar(im, ax=ax0, fraction=0.025, pad=0.02)
    cbar.set_label("mean(|weight|)")

    order = np.argsort(channel_mean)[::-1]
    ax1.barh(np.arange(len(CHANNEL_NAMES)), channel_mean[order], color="#c44e52")
    ax1.set_yticks(np.arange(len(CHANNEL_NAMES)))
    ax1.set_yticklabels(np.array(CHANNEL_NAMES)[order], fontsize=8.5)
    ax1.invert_yaxis()
    ax1.set_xlabel("Average fusion strength")
    ax1.set_title("Channel ranking")

    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.24, top=0.88)
    fig.suptitle("How the first CNN layer combines the 21 EEG channels", fontsize=16, y=0.95)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def save_saliency_figure(
    checkpoint_path: Path,
    master_index: pd.DataFrame,
    seizure_case: pd.Series,
    output_path: Path,
) -> None:
    set_research_style()
    ensure_dir(output_path.parent)

    raw_window, row = load_window_from_row(master_index, int(seizure_case["row_id"]))
    model, mean, std, threshold = load_trained_model(checkpoint_path, device="cpu")

    normalized = (raw_window - mean[:, None]) / std[:, None]
    x = torch.tensor(normalized[None, ...], dtype=torch.float32, requires_grad=True)
    logits = model(x)
    logits.backward(torch.ones_like(logits))
    saliency = x.grad.detach().abs().cpu().numpy()[0]
    channel_importance = saliency.mean(axis=1)
    top_idx = np.argsort(channel_importance)[-6:][::-1]

    raw_norm = _signal_norm(raw_window)
    sal_norm = colors.Normalize(vmin=0.0, vmax=float(np.percentile(saliency, 99)))

    fig = plt.figure(figsize=(18, 11.5))
    gs = GridSpec(2, 2, height_ratios=[1.0, 1.0], width_ratios=[4.0, 1.35], hspace=0.30, wspace=0.20, figure=fig)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[:, 1])

    im0 = ax0.imshow(raw_window, aspect="auto", cmap="RdBu_r", norm=raw_norm, interpolation="nearest")
    ax0.set_title("Raw EEG window")
    ax0.set_ylabel("EEG channel")
    ax0.set_yticks(np.arange(len(CHANNEL_NAMES)))
    ax0.set_yticklabels(CHANNEL_NAMES, fontsize=8)
    ax0.set_xticks([])
    cbar0 = fig.colorbar(im0, ax=ax0, fraction=0.025, pad=0.02)
    cbar0.set_label("Amplitude")

    im1 = ax1.imshow(saliency, aspect="auto", cmap="viridis", norm=sal_norm, interpolation="nearest")
    ax1.set_title("Gradient saliency map |d logit / d input|")
    ax1.set_xlabel("Time index (128 samples)")
    ax1.set_ylabel("EEG channel")
    ax1.set_yticks(np.arange(len(CHANNEL_NAMES)))
    ax1.set_yticklabels(CHANNEL_NAMES, fontsize=8)
    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.025, pad=0.02)
    cbar1.set_label("Saliency")

    ax2.barh(np.arange(len(top_idx)), channel_importance[top_idx], color="#4c72b0")
    ax2.set_yticks(np.arange(len(top_idx)))
    ax2.set_yticklabels(np.array(CHANNEL_NAMES)[top_idx])
    ax2.invert_yaxis()
    ax2.set_xlabel("Mean saliency")
    ax2.set_title("Top channels for this seizure example")

    score = float(seizure_case["y_score"])
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.08, top=0.88)
    fig.suptitle("Saliency-based explanation for a representative seizure example", fontsize=16, y=0.95)
    fig.text(
        0.5,
        0.905,
        f"row_id={int(seizure_case['row_id'])} | patient={row['patient_id']} | score={score:.3f} | threshold={threshold:.3f}",
        ha="center",
        va="center",
        fontsize=10,
        color="#444444",
    )
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def build_research_visual_bundle(project_root: Path = PROJECT_ROOT) -> dict[str, object]:
    output_dir = research_visuals_dir(project_root)
    master_index = load_master_index(project_root / "fold1_data_intake" / "master_index.parquet")
    prediction_path = reference_prediction_path(project_root)
    seizure_case, normal_case, checkpoint_path = select_reference_cases(master_index, prediction_path)
    seizure_window, seizure_row = load_window_from_row(master_index, int(seizure_case["row_id"]))
    normal_window, normal_row = load_window_from_row(master_index, int(normal_case["row_id"]))

    input_path = output_dir / "input_heatmap_pair.png"
    arch_path = output_dir / "channel_fusion_architecture.png"
    weight_path = output_dir / "first_layer_channel_fusion_weights.png"
    saliency_path = output_dir / "saliency_case_patient_fold0.png"
    notes_path = output_dir / "research_visuals_notes.md"
    captions_zh_path = output_dir / "research_visuals_captions_zh.md"
    manifest_path = output_dir / "research_visuals_manifest.json"

    save_input_heatmap_pair(normal_window, seizure_window, normal_case, seizure_case, input_path)
    save_channel_fusion_architecture(arch_path)
    save_first_layer_weight_heatmap(checkpoint_path, weight_path)
    save_saliency_figure(checkpoint_path, master_index, seizure_case, saliency_path)

    notes = [
        "# Research-style Visualizations",
        "",
        "## 1. Input Heatmap Pair",
        "- Compares a representative non-seizure window and a representative seizure window from the same patient-holdout test fold.",
        "- Shows the raw 21-channel by 128-sample matrix that enters the CNN.",
        "",
        "## 2. Channel Fusion Architecture",
        "- Explains that the baseline is a single-window classifier with input-level channel fusion.",
        "- The first Conv1d layer already mixes all 21 EEG channels.",
        "",
        "## 3. First-layer Weight Heatmap",
        "- Summarizes how strongly each first-layer filter uses each EEG channel.",
        "- Useful to justify that channel fusion is actually learned, not only claimed.",
        "",
        "## 4. Saliency Figure",
        "- Shows which channels and time regions contributed most to a seizure decision for one representative true-positive test example.",
        "",
        "## Reference Slice",
        f"- Protocol: `{REFERENCE_PROTOCOL}`",
        f"- Train mode: `{REFERENCE_TRAIN_MODE}`",
        f"- Test mode: `{REFERENCE_TEST_MODE}`",
        f"- Outer fold: `{REFERENCE_OUTER_FOLD:03d}`",
        f"- Checkpoint: `{checkpoint_path}`",
    ]
    notes_path.write_text("\n".join(notes) + "\n", encoding="utf-8")

    captions_zh = [
        "# 中文图注",
        "",
        "## 图 1. 单窗口 EEG 输入热力图对比",
        "左图展示一个代表性的非发作窗口，右图展示一个代表性的发作窗口。每个窗口均由 `21` 个 EEG 通道和 `128` 个时间采样点组成。该图用于说明本研究的 CNN 基线并不是对单通道分别分类，而是直接接收完整的多通道 EEG 矩阵作为输入。",
        "",
        "## 图 2. Channel Fusion CNN 结构示意图",
        "该图展示了当前基线模型的输入级通道融合流程。原始 EEG 窗口以 `[21, 128]` 的多通道时序矩阵进入网络，在第一层 `Conv1d(21 -> 16)` 中，不同卷积核已经开始联合建模全部 EEG 通道，因此该模型已经实现了 channel fusion。",
        "",
        "## 图 3. 第一层卷积核的通道融合权重热图",
        "热图展示第一层卷积核对不同 EEG 通道的平均权重强度，右侧条形图给出各通道的整体重要性排序。该图提供了全局层面的证据，说明模型确实在学习跨通道融合，而不是只依赖个别通道的局部信息。",
        "",
        "## 图 4. 发作样本的 Saliency 可解释性结果",
        "上图为原始 EEG 发作窗口，下图为对应的梯度显著性热图，右侧条形图给出该样本中最重要的通道。该图提供了局部层面的证据，说明模型在做出发作判别时，重点关注的是哪些通道以及哪些时间片段。",
    ]
    captions_zh_path.write_text("\n".join(captions_zh) + "\n", encoding="utf-8")

    payload = {
        "reference_protocol": REFERENCE_PROTOCOL,
        "reference_train_mode": REFERENCE_TRAIN_MODE,
        "reference_test_mode": REFERENCE_TEST_MODE,
        "reference_outer_fold_id": REFERENCE_OUTER_FOLD,
        "prediction_path": str(prediction_path),
        "checkpoint_path": str(checkpoint_path),
        "representative_normal_case": {
            "row_id": int(normal_case["row_id"]),
            "patient_id": str(normal_row["patient_id"]),
            "filename": str(normal_row["filename"]),
            "global_interval": int(normal_row["global_interval"]),
            "y_score": float(normal_case["y_score"]),
        },
        "representative_seizure_case": {
            "row_id": int(seizure_case["row_id"]),
            "patient_id": str(seizure_row["patient_id"]),
            "filename": str(seizure_row["filename"]),
            "global_interval": int(seizure_row["global_interval"]),
            "y_score": float(seizure_case["y_score"]),
        },
        "artifacts": {
            "input_heatmap_pair": str(input_path),
            "channel_fusion_architecture": str(arch_path),
            "first_layer_channel_fusion_weights": str(weight_path),
            "saliency_case": str(saliency_path),
            "notes": str(notes_path),
            "captions_zh": str(captions_zh_path),
        },
    }
    write_json(manifest_path, payload)
    return payload
