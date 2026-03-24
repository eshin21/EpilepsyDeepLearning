from __future__ import annotations

import json
import math
import os
import zipfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA_ROOT = PROJECT_ROOT / "data" / "epilepsy"
FOLD1_DIR = PROJECT_ROOT / "fold1_data_intake"
FOLD2_DIR = PROJECT_ROOT / "fold2_data_audit"
FOLD3_DIR = PROJECT_ROOT / "fold3_split_protocols"
FOLD4_DIR = PROJECT_ROOT / "fold4_input_pipeline"
FOLD5_DIR = PROJECT_ROOT / "fold5_cnn_training"
FOLD6_DIR = PROJECT_ROOT / "fold6_evaluation"
FOLD7_DIR = PROJECT_ROOT / "fold7_results_and_reporting"
FOLD8_DIR = PROJECT_ROOT / "fold8_future_slot"
CACHE_DIR = FOLD4_DIR / "patient_cache"
SIGNAL_KEY = "EEG_win"
SIGNAL_SHAPE = (21, 128)
SEED = 2026


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _yaml_lines(payload: Any, indent: int = 0) -> list[str]:
    pad = "  " * indent
    if isinstance(payload, dict):
        lines: list[str] = []
        for key, value in payload.items():
            if isinstance(value, (dict, list)):
                lines.append(f"{pad}{key}:")
                lines.extend(_yaml_lines(value, indent + 1))
            else:
                lines.append(f"{pad}{key}: {value}")
        return lines
    if isinstance(payload, list):
        lines = []
        for value in payload:
            if isinstance(value, (dict, list)):
                lines.append(f"{pad}-")
                lines.extend(_yaml_lines(value, indent + 1))
            else:
                lines.append(f"{pad}- {value}")
        return lines
    return [f"{pad}{payload}"]


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(_yaml_lines(payload)) + "\n")


def patient_id_from_filename(name: str) -> str:
    return name.split("_")[0]


def patient_paths(data_root: Path = DATA_ROOT) -> list[tuple[str, Path, Path]]:
    patients: list[tuple[str, Path, Path]] = []
    for meta_path in sorted(data_root.glob("chb*_seizure_metadata_1.parquet")):
        patient_id = meta_path.name.split("_")[0]
        npz_path = data_root / f"{patient_id}_seizure_EEGwindow_1.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"Missing signal file for {patient_id}: {npz_path}")
        patients.append((patient_id, meta_path, npz_path))
    if not patients:
        raise FileNotFoundError(f"No patient metadata files found under {data_root}")
    return patients


def read_npz_header(npz_path: Path, array_key: str = SIGNAL_KEY) -> tuple[tuple[int, ...], bool, str]:
    with zipfile.ZipFile(npz_path, "r") as archive:
        entry_name = f"{array_key}.npy"
        if entry_name not in archive.namelist():
            raise KeyError(f"{entry_name} not found inside {npz_path}")
        with archive.open(entry_name, "r") as handle:
            version = np.lib.format.read_magic(handle)
            if version == (1, 0):
                shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(handle)
            else:
                shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(handle)
    return shape, fortran_order, str(dtype)


def load_raw_npz_array(npz_path: Path, array_key: str = SIGNAL_KEY) -> np.ndarray:
    with np.load(npz_path, allow_pickle=True) as payload:
        return payload[array_key]


def load_single_window(npz_path: Path, window_idx: int, array_key: str = SIGNAL_KEY) -> np.ndarray:
    raw = load_raw_npz_array(npz_path=npz_path, array_key=array_key)
    return np.asarray(raw[window_idx], dtype=np.float32)


def _build_inventory_row(master_slice: pd.DataFrame) -> dict[str, Any]:
    return {
        "patient_id": master_slice["patient_id"].iloc[0],
        "n_windows": int(len(master_slice)),
        "n_pos_windows": int((master_slice["class_label"] == 1).sum()),
        "n_neg_windows": int((master_slice["class_label"] == 0).sum()),
        "n_files": int(master_slice["filename"].nunique()),
        "n_events": int(master_slice.loc[master_slice["class_label"] == 1, "global_interval"].nunique()),
    }


def build_master_index(data_root: Path = DATA_ROOT, force: bool = False) -> Path:
    ensure_dir(FOLD1_DIR)
    master_index_path = FOLD1_DIR / "master_index.parquet"
    inventory_path = FOLD1_DIR / "patient_inventory.csv"
    integrity_path = FOLD1_DIR / "integrity_report.json"
    contract_path = FOLD1_DIR / "data_contract.md"

    if master_index_path.exists() and inventory_path.exists() and integrity_path.exists() and not force:
        return master_index_path

    frames: list[pd.DataFrame] = []
    inventory_rows: list[dict[str, Any]] = []
    row_id_offset = 0
    mismatches: list[dict[str, Any]] = []

    for patient_id, meta_path, npz_path in patient_paths(data_root):
        meta = pd.read_parquet(meta_path).copy()
        shape, fortran_order, dtype = read_npz_header(npz_path)
        if tuple(shape[1:]) != SIGNAL_SHAPE:
            raise ValueError(f"Unexpected signal shape for {patient_id}: {shape}")
        row_count_matches = shape[0] == len(meta)
        if not row_count_matches:
            mismatches.append(
                {
                    "patient_id": patient_id,
                    "metadata_rows": int(len(meta)),
                    "signal_rows": int(shape[0]),
                }
            )
        meta.insert(0, "row_id", np.arange(row_id_offset, row_id_offset + len(meta), dtype=np.int64))
        meta.insert(1, "patient_id", patient_id)
        meta.rename(columns={"class": "class_label"}, inplace=True)
        meta["window_idx_in_patient"] = np.arange(len(meta), dtype=np.int64)
        meta["source_npz_path"] = str(npz_path)
        meta["signal_key"] = SIGNAL_KEY
        meta["signal_shape"] = f"{SIGNAL_SHAPE[0]}x{SIGNAL_SHAPE[1]}"
        meta["npz_header_dtype"] = dtype
        meta["npz_fortran_order"] = bool(fortran_order)
        frames.append(meta)
        inventory_rows.append(_build_inventory_row(meta))
        row_id_offset += len(meta)

    master = pd.concat(frames, ignore_index=True)
    inventory = pd.DataFrame(inventory_rows).sort_values("patient_id").reset_index(drop=True)
    master.to_parquet(master_index_path, index=False)
    inventory.to_csv(inventory_path, index=False)

    integrity_report = {
        "data_root": str(data_root),
        "master_index_path": str(master_index_path),
        "n_patients": int(master["patient_id"].nunique()),
        "n_rows": int(len(master)),
        "n_pos_windows": int((master["class_label"] == 1).sum()),
        "n_neg_windows": int((master["class_label"] == 0).sum()),
        "row_id_unique": bool(master["row_id"].is_unique),
        "signal_key": SIGNAL_KEY,
        "signal_shape": list(SIGNAL_SHAPE),
        "mismatches": mismatches,
    }
    write_json(integrity_path, integrity_report)

    contract_lines = [
        "# Data Contract",
        "",
        f"- Raw data symlink: `{data_root}`",
        f"- Master index: `{master_index_path}`",
        f"- Expected signal key: `{SIGNAL_KEY}`",
        f"- Expected window shape: `{SIGNAL_SHAPE}`",
        f"- Patients discovered: `{integrity_report['n_patients']}`",
        f"- Windows discovered: `{integrity_report['n_rows']}`",
        "",
        "## Master Index Columns",
        "",
        "- `row_id`: global unique window id",
        "- `patient_id`: subject identifier",
        "- `class_label`: 0 normal / 1 seizure",
        "- `filename_interval`: recording-local interval id",
        "- `global_interval`: patient-level seizure/event id",
        "- `filename`: EDF filename",
        "- `window_idx_in_patient`: row index inside the patient signal tensor",
        "- `source_npz_path`: raw window file",
        "- `signal_key`: array key inside the npz",
        "- `signal_shape`: textual `21x128` marker",
    ]
    contract_path.write_text("\n".join(contract_lines) + "\n", encoding="utf-8")
    return master_index_path


def load_master_index(master_index_path: Path | None = None) -> pd.DataFrame:
    path = master_index_path or FOLD1_DIR / "master_index.parquet"
    if not path.exists():
        build_master_index(force=False)
    return pd.read_parquet(path)


def normal_only_recordings(master_index: pd.DataFrame) -> pd.DataFrame:
    file_flags = (
        master_index.groupby(["patient_id", "filename"])["class_label"]
        .max()
        .reset_index(name="has_positive")
        .sort_values(["patient_id", "filename"])
    )
    return file_flags.loc[file_flags["has_positive"] == 0, ["patient_id", "filename"]].reset_index(drop=True)


def normal_only_intervals(master_index: pd.DataFrame) -> pd.DataFrame:
    interval_flags = (
        master_index.groupby(["patient_id", "global_interval"])["class_label"]
        .max()
        .reset_index(name="has_positive")
        .sort_values(["patient_id", "global_interval"])
    )
    return interval_flags.loc[interval_flags["has_positive"] == 0, ["patient_id", "global_interval"]].reset_index(drop=True)


def materialize_patient_cache(
    patient_id: str,
    npz_path: Path,
    cache_dir: Path = CACHE_DIR,
    force: bool = False,
    chunk_size: int = 512,
) -> Path:
    ensure_dir(cache_dir)
    cache_path = cache_dir / f"{patient_id}_windows.float32.npy"
    meta_path = cache_dir / f"{patient_id}_windows.meta.json"
    if cache_path.exists() and meta_path.exists() and not force:
        return cache_path

    raw = load_raw_npz_array(npz_path)
    shape = raw.shape
    if tuple(shape[1:]) != SIGNAL_SHAPE:
        raise ValueError(f"Unexpected raw cache shape for {patient_id}: {shape}")
    memmap = np.lib.format.open_memmap(cache_path, mode="w+", dtype=np.float32, shape=shape)
    for start in range(0, shape[0], chunk_size):
        stop = min(start + chunk_size, shape[0])
        block = [np.asarray(window, dtype=np.float32) for window in raw[start:stop]]
        memmap[start:stop] = np.stack(block, axis=0)
    memmap.flush()
    write_json(
        meta_path,
        {
            "patient_id": patient_id,
            "cache_path": str(cache_path),
            "shape": list(shape),
            "dtype": "float32",
            "source_npz_path": str(npz_path),
        },
    )
    return cache_path


def ensure_patient_caches(
    master_index: pd.DataFrame,
    cache_dir: Path = CACHE_DIR,
    patient_ids: list[str] | None = None,
    force: bool = False,
) -> dict[str, Path]:
    ensure_dir(cache_dir)
    if patient_ids is None:
        patient_ids = sorted(master_index["patient_id"].unique().tolist())
    patient_sources = (
        master_index[["patient_id", "source_npz_path"]]
        .drop_duplicates()
        .sort_values("patient_id")
        .set_index("patient_id")["source_npz_path"]
        .to_dict()
    )
    cache_paths: dict[str, Path] = {}
    for patient_id in patient_ids:
        npz_path = Path(patient_sources[patient_id])
        cache_paths[patient_id] = materialize_patient_cache(patient_id, npz_path, cache_dir=cache_dir, force=force)
    return cache_paths


def build_audit_artifacts(master_index: pd.DataFrame, force: bool = False) -> None:
    ensure_dir(FOLD2_DIR)
    patient_stats_path = FOLD2_DIR / "patient_stats.csv"
    class_ratio_path = FOLD2_DIR / "class_ratio.csv"
    event_stats_path = FOLD2_DIR / "event_stats.csv"
    summary_path = FOLD2_DIR / "eda_summary.md"
    if patient_stats_path.exists() and class_ratio_path.exists() and event_stats_path.exists() and summary_path.exists() and not force:
        return

    patient_stats = (
        master_index.groupby("patient_id")
        .agg(
            n_windows=("row_id", "count"),
            n_pos_windows=("class_label", lambda s: int((s == 1).sum())),
            n_neg_windows=("class_label", lambda s: int((s == 0).sum())),
            n_files=("filename", "nunique"),
        )
        .reset_index()
        .sort_values("patient_id")
    )
    event_counts = (
        master_index.loc[master_index["class_label"] == 1]
        .groupby("patient_id")["global_interval"]
        .nunique()
        .rename("n_events")
        .reset_index()
    )
    patient_stats = patient_stats.merge(event_counts, on="patient_id", how="left").fillna({"n_events": 0})
    patient_stats["n_events"] = patient_stats["n_events"].astype(int)
    class_ratio = pd.DataFrame(
        [
            {
                "label": "normal",
                "class_label": 0,
                "n_windows": int((master_index["class_label"] == 0).sum()),
            },
            {
                "label": "seizure",
                "class_label": 1,
                "n_windows": int((master_index["class_label"] == 1).sum()),
            },
        ]
    )
    class_ratio["ratio"] = class_ratio["n_windows"] / float(class_ratio["n_windows"].sum())

    event_stats = (
        master_index.groupby(["patient_id", "global_interval"])
        .agg(
            n_windows=("row_id", "count"),
            n_pos_windows=("class_label", lambda s: int((s == 1).sum())),
            n_neg_windows=("class_label", lambda s: int((s == 0).sum())),
            files=("filename", "nunique"),
        )
        .reset_index()
        .sort_values(["patient_id", "global_interval"])
    )

    patient_stats.to_csv(patient_stats_path, index=False)
    class_ratio.to_csv(class_ratio_path, index=False)
    event_stats.to_csv(event_stats_path, index=False)

    _plot_bar(
        patient_stats["patient_id"].tolist(),
        patient_stats["n_windows"].tolist(),
        FOLD2_DIR / "windows_per_patient.png",
        title="Windows per patient",
        ylabel="Number of windows",
    )
    _plot_bar(
        patient_stats["patient_id"].tolist(),
        patient_stats["n_events"].tolist(),
        FOLD2_DIR / "seizures_per_patient.png",
        title="Seizure events per patient",
        ylabel="Number of seizure events",
    )
    _plot_histogram(
        event_stats["n_pos_windows"].tolist(),
        FOLD2_DIR / "interval_length_distribution.png",
        title="Positive windows per event",
        xlabel="Positive windows",
        ylabel="Frequency",
    )
    _plot_sample_windows(master_index, FOLD2_DIR / "sample_windows.png")

    summary_lines = [
        "# EDA Summary",
        "",
        f"- Total patients: `{int(master_index['patient_id'].nunique())}`",
        f"- Total windows: `{int(len(master_index))}`",
        f"- Total seizure windows: `{int((master_index['class_label'] == 1).sum())}`",
        f"- Total normal windows: `{int((master_index['class_label'] == 0).sum())}`",
        f"- Total seizure events: `{int(master_index.loc[master_index['class_label'] == 1, 'global_interval'].nunique())}` within-patient unique ids",
        "",
        "## Key observations",
        "",
        "- The dataset is strongly imbalanced toward normal windows.",
        "- Event counts vary substantially across patients.",
        "- Window-level validation should be interpreted as an upper bound because train and test can share patient/event context.",
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


def _style_publication_axes(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#6b7280")
    ax.spines["bottom"].set_color("#6b7280")
    ax.tick_params(colors="#374151")
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, linestyle="--", linewidth=0.6, alpha=0.28, color="#9aa5b1")


def _plot_bar(labels: list[str], values: list[int], output_path: Path, title: str, ylabel: str) -> None:
    palette = {
        "windows_per_patient.png": "#4C78A8",
        "seizures_per_patient.png": "#54A24B",
    }
    bar_color = palette.get(output_path.name, "#4C78A8")
    fig, ax = plt.subplots(figsize=(12.5, 4.6))
    positions = np.arange(len(labels))
    ax.bar(positions, values, color=bar_color, edgecolor="#31445b", linewidth=0.45, alpha=0.92)
    _style_publication_axes(ax, grid_axis="y")
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color="#1f2937")
    ax.set_ylabel(ylabel)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=90)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, facecolor="white")
    plt.close(fig)


def _plot_histogram(values: list[int], output_path: Path, title: str, xlabel: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    bins = min(30, max(10, int(math.sqrt(max(len(values), 1)))))
    ax.hist(values, bins=bins, color="#DD8452", edgecolor="#5B4636", linewidth=0.5, alpha=0.88)
    _style_publication_axes(ax, grid_axis="y")
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color="#1f2937")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, facecolor="white")
    plt.close(fig)


def _plot_sample_windows(master_index: pd.DataFrame, output_path: Path) -> None:
    positive_row = master_index.loc[master_index["class_label"] == 1].iloc[0]
    negative_row = master_index.loc[master_index["class_label"] == 0].iloc[0]
    positive_window = load_single_window(Path(positive_row["source_npz_path"]), int(positive_row["window_idx_in_patient"]))
    negative_window = load_single_window(Path(negative_row["source_npz_path"]), int(negative_row["window_idx_in_patient"]))
    merged = np.concatenate([negative_window.reshape(-1), positive_window.reshape(-1)])
    vmax = float(np.percentile(np.abs(merged), 99))
    vmax = max(vmax, 1.0)
    norm = matplotlib.colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), sharey=True)
    for idx, (ax, window, title) in enumerate([
        (axes[0], negative_window, "Normal window"),
        (axes[1], positive_window, "Seizure window"),
    ]):
        im = ax.imshow(window, aspect="auto", cmap="cividis", norm=norm, interpolation="nearest")
        ax.set_title(title, fontsize=12.5, fontweight="bold", color="#1f2937")
        ax.set_xlabel("Time index")
        if idx == 0:
            ax.set_ylabel("EEG channel")
        else:
            ax.set_ylabel("")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#6b7280")
        ax.spines["bottom"].set_color("#6b7280")
        ax.tick_params(colors="#374151")
        fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, facecolor="white")
    plt.close(fig)
