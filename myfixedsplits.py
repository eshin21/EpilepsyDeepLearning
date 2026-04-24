from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from .data_io import FOLD3_DIR, SEED, ensure_dir, normal_only_intervals, normal_only_recordings


WINDOW_FOLDS = 15
VAL_FRACTION = 0.10
TRAIN_MODE_TO_RATIO = {
    "balanced_50_50": 0.50,
    "unbalanced_20_80": 0.20,
}
TEST_MODE_TO_RATIO = TRAIN_MODE_TO_RATIO.copy()


@dataclass
class FoldPools:
    train_pool: pd.DataFrame
    val_pool: pd.DataFrame
    test_pool: pd.DataFrame
    metadata: dict[str, object]


def _preprocess_master_index(master_index: pd.DataFrame) -> pd.DataFrame:
    """Merges chb21 into chb01 to prevent patient-level data leakage."""
    df = master_index.copy()
    chb21_mask = df["patient_id"] == "chb21"
    df.loc[chb21_mask, "global_interval"] += 100  # Shift interval IDs to avoid collisions
    df.loc[chb21_mask, "patient_id"] = "chb01"
    return df


def build_split_artifacts(master_index: pd.DataFrame, force: bool = False) -> dict[str, Path]:
    ensure_dir(FOLD3_DIR)
    window_path = FOLD3_DIR / "window_manifest.parquet"
    seizure_path = FOLD3_DIR / "seizure_manifest.parquet"
    patient_path = FOLD3_DIR / "patient_manifest.parquet"
    normal_path = FOLD3_DIR / "normal_only_recordings.parquet"
    interval_path = FOLD3_DIR / "normal_only_intervals.parquet"
    report_path = FOLD3_DIR / "split_qc_report.md"
    overlap_path = FOLD3_DIR / "window_overlap_summary.csv"

    if all(path.exists() for path in[window_path, seizure_path, patient_path, normal_path, interval_path, report_path, overlap_path]) and not force:
        return {
            "window": window_path,
            "seizure": seizure_path,
            "patient": patient_path,
            "normal_only_recordings": normal_path,
            "normal_only_intervals": interval_path,
            "report": report_path,
            "overlap": overlap_path,
        }

    # APPLY PATIENT MERGE FIX
    df = _preprocess_master_index(master_index)

    window_assignments = _build_window_assignments(df)
    seizure_assignments = _build_seizure_assignments(df)
    patient_assignments = _build_patient_assignments(df)
    normal_only = normal_only_recordings(df)
    normal_interval_only = normal_only_intervals(df)

    window_assignments.to_parquet(window_path, index=False)
    seizure_assignments.to_parquet(seizure_path, index=False)
    patient_assignments.to_parquet(patient_path, index=False)
    normal_only.to_parquet(normal_path, index=False)
    normal_interval_only.to_parquet(interval_path, index=False)

    overlap_rows =[]
    for fold_id in sorted(window_assignments["outer_fold_id"].unique().tolist()):
        test_rows = df.loc[window_assignments["outer_fold_id"] == fold_id]
        train_rows = df.loc[window_assignments["outer_fold_id"] != fold_id]
        overlap_rows.append(
            {
                "outer_fold_id": int(fold_id),
                "shared_patients": int(
                    len(set(train_rows["patient_id"].unique()).intersection(set(test_rows["patient_id"].unique())))
                ),
                "shared_positive_events": int(
                    len(
                        set(
                            zip(
                                train_rows.loc[train_rows["class_label"] == 1, "patient_id"],
                                train_rows.loc[train_rows["class_label"] == 1, "global_interval"],
                            )
                        ).intersection(
                            set(
                                zip(
                                    test_rows.loc[test_rows["class_label"] == 1, "patient_id"],
                                    test_rows.loc[test_rows["class_label"] == 1, "global_interval"],
                                )
                            )
                        )
                    )
                ),
            }
        )
    overlap_df = pd.DataFrame(overlap_rows)
    overlap_df.to_csv(overlap_path, index=False)

    report_lines =[
        "# Split QC Report",
        "",
        f"- Window folds: `{window_assignments['outer_fold_id'].nunique()}`",
        f"- Seizure folds: `{len(seizure_assignments)}`",
        f"- Patient folds: `{len(patient_assignments)}`",
        f"- Normal-only recordings: `{len(normal_only)}`",
        f"- Normal-only intervals: `{len(normal_interval_only)}`",
        "",
        "## Notes",
        "",
        "- Window folds exclude +/- 4 adjacent overlapping seizure windows to prevent 80% overlap leakage.",
        "- Seizure folds are keyed by `(patient_id, global_interval)` and prefer normal-only recordings for negatives.",
        "- Patient folds merge chb21 and chb01 to prevent unseen-patient leakage.",
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    return {
        "window": window_path,
        "seizure": seizure_path,
        "patient": patient_path,
        "normal_only_recordings": normal_path,
        "normal_only_intervals": interval_path,
        "report": report_path,
        "overlap": overlap_path,
    }


def resolve_fold_pools(
    master_index: pd.DataFrame,
    protocol: str,
    outer_fold_id: int,
    seed: int = SEED,
    val_fraction: float = VAL_FRACTION,
) -> FoldPools:
    df = _preprocess_master_index(master_index)
    split_paths = build_split_artifacts(master_index, force=False)
    if protocol == "window":
        assignments = pd.read_parquet(split_paths["window"])
        return _resolve_window_fold(df, assignments, outer_fold_id, seed=seed, val_fraction=val_fraction)
    if protocol == "seizure":
        assignments = pd.read_parquet(split_paths["seizure"])
        normal_only = pd.read_parquet(split_paths["normal_only_recordings"])
        normal_intervals = pd.read_parquet(split_paths["normal_only_intervals"])
        return _resolve_seizure_fold(
            df,
            assignments,
            normal_only,
            normal_intervals,
            outer_fold_id,
            seed=seed,
            val_fraction=val_fraction,
        )
    if protocol == "patient":
        assignments = pd.read_parquet(split_paths["patient"])
        return _resolve_patient_fold(df, assignments, outer_fold_id, seed=seed, val_fraction=val_fraction)
    raise ValueError(f"Unsupported protocol: {protocol}")


def sample_rows_for_ratio(
    rows: pd.DataFrame,
    target_positive_ratio: float,
    seed: int,
    label_column: str = "class_label",
) -> pd.DataFrame:
    positive = rows.loc[rows[label_column] == 1]
    negative = rows.loc[rows[label_column] == 0]
    if positive.empty or negative.empty:
        return rows.copy().reset_index(drop=True)

    rng = np.random.default_rng(seed)
    if abs(target_positive_ratio - 0.5) < 1e-9:
        n_pos = min(len(positive), len(negative))
        n_neg = n_pos
    elif abs(target_positive_ratio - 0.2) < 1e-9:
        units = min(len(positive), len(negative) // 4)
        if units == 0:
            return rows.copy().reset_index(drop=True)
        n_pos = units
        n_neg = units * 4
    else:
        raise ValueError(f"Unsupported target ratio: {target_positive_ratio}")

    pos_sample = positive.sample(n=n_pos, replace=False, random_state=int(rng.integers(0, 1_000_000)))
    neg_sample = negative.sample(n=n_neg, replace=False, random_state=int(rng.integers(0, 1_000_000)))
    sampled = pd.concat([pos_sample, neg_sample], ignore_index=True)
    sampled = sampled.sample(frac=1.0, random_state=int(rng.integers(0, 1_000_000))).reset_index(drop=True)
    return sampled


def smoke_subset(rows: pd.DataFrame, seed: int, max_positive: int = 128, max_negative: int = 128) -> pd.DataFrame:
    positive = rows.loc[rows["class_label"] == 1]
    negative = rows.loc[rows["class_label"] == 0]
    if len(positive) > max_positive:
        positive = positive.sample(n=max_positive, random_state=seed)
    if len(negative) > max_negative:
        negative = negative.sample(n=max_negative, random_state=seed + 1)
    return pd.concat([positive, negative], ignore_index=True).sample(frac=1.0, random_state=seed + 2).reset_index(drop=True)


def list_outer_fold_ids(master_index: pd.DataFrame, protocol: str) -> list[int]:
    split_paths = build_split_artifacts(master_index, force=False)
    assignments = pd.read_parquet(split_paths[protocol])
    return sorted(assignments["outer_fold_id"].unique().tolist())


def _build_window_assignments(df: pd.DataFrame) -> pd.DataFrame:
    assignments = pd.DataFrame({"row_id": df["row_id"].to_numpy(), "outer_fold_id": -1})
    splitter = StratifiedKFold(n_splits=WINDOW_FOLDS, shuffle=True, random_state=SEED)
    labels = df["class_label"].to_numpy()
    for fold_id, (_, test_idx) in enumerate(splitter.split(df["row_id"].to_numpy(), labels)):
        assignments.loc[test_idx, "outer_fold_id"] = fold_id
    assignments["group_id"] = assignments["row_id"].astype(str)
    assignments["protocol"] = "window"
    return assignments


def _build_seizure_assignments(df: pd.DataFrame) -> pd.DataFrame:
    seizure_rows = df.loc[df["class_label"] == 1, ["patient_id", "global_interval"]].drop_duplicates()
    seizure_rows = seizure_rows.sort_values(["patient_id", "global_interval"]).reset_index(drop=True)
    seizure_rows["outer_fold_id"] = np.arange(len(seizure_rows), dtype=np.int64)
    seizure_rows["group_id"] = seizure_rows["patient_id"] + "__" + seizure_rows["global_interval"].astype(str)
    seizure_rows["protocol"] = "seizure"
    return seizure_rows


def _build_patient_assignments(df: pd.DataFrame) -> pd.DataFrame:
    patient_rows = (
        df[["patient_id"]]
        .drop_duplicates()
        .sort_values("patient_id")
        .reset_index(drop=True)
    )
    patient_rows["outer_fold_id"] = np.arange(len(patient_rows), dtype=np.int64)
    patient_rows["group_id"] = patient_rows["patient_id"]
    patient_rows["protocol"] = "patient"
    return patient_rows


def _resolve_window_fold(
    df: pd.DataFrame,
    assignments: pd.DataFrame,
    outer_fold_id: int,
    seed: int,
    val_fraction: float,
) -> FoldPools:
    test_ids = assignments.loc[assignments["outer_fold_id"] == outer_fold_id, "row_id"].values
    test_pool = df.loc[df["row_id"].isin(test_ids)].copy()
    
    # EXACT FIX: EXCLUDE +/- 4 LEAKAGE NEIGHBORS **ONLY FOR SEIZURE WINDOWS**
    test_pos_ids = test_pool.loc[test_pool["class_label"] == 1, "row_id"].values
    exclude_ids = set(test_ids)
    for offset in range(1, 5):
        exclude_ids.update(test_pos_ids - offset)
        exclude_ids.update(test_pos_ids + offset)
        
    train_val = df.loc[~df["row_id"].isin(exclude_ids)].copy()
    
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed + outer_fold_id)
    indices = next(splitter.split(train_val["row_id"].to_numpy(), train_val["class_label"].to_numpy()))
    train_idx, val_idx = indices
    
    train_pool_temp = train_val.iloc[train_idx].copy()
    val_pool = train_val.iloc[val_idx].copy()
    
    # ALSO PREVENT LEAKAGE BETWEEN VAL AND TRAIN (ONLY FOR SEIZURE WINDOWS)
    val_pos_ids = val_pool.loc[val_pool["class_label"] == 1, "row_id"].values
    val_exclude_ids = set(val_pool["row_id"].values)
    for offset in range(1, 5):
        val_exclude_ids.update(val_pos_ids - offset)
        val_exclude_ids.update(val_pos_ids + offset)
        
    train_pool = train_pool_temp.loc[~train_pool_temp["row_id"].isin(val_exclude_ids)].copy()

    return FoldPools(
        train_pool=train_pool.reset_index(drop=True),
        val_pool=val_pool.reset_index(drop=True),
        test_pool=test_pool.reset_index(drop=True),
        metadata={
            "protocol": "window",
            "outer_fold_id": int(outer_fold_id),
            "test_group_id": f"window_fold_{outer_fold_id:02d}",
        },
    )


def _resolve_seizure_fold(
    df: pd.DataFrame,
    assignments: pd.DataFrame,
    normal_only: pd.DataFrame,
    normal_intervals: pd.DataFrame,
    outer_fold_id: int,
    seed: int,
    val_fraction: float,
) -> FoldPools:
    held_out = assignments.loc[assignments["outer_fold_id"] == outer_fold_id].iloc[0]
    test_patient = held_out["patient_id"]
    test_interval = int(held_out["global_interval"])

    event_mask = (df["patient_id"] == test_patient) & (df["global_interval"] == test_interval)
    test_pos_pool = df.loc[event_mask & (df["class_label"] == 1)].copy()
    test_neg_pool = _select_seizure_negative_pool(df, normal_only, normal_intervals, test_patient, exclude_intervals={test_interval})

    remaining_events = assignments.loc[(assignments["outer_fold_id"] != outer_fold_id) & (assignments["patient_id"] != test_patient)].copy()
    n_val_events = max(1, int(round(len(remaining_events) * val_fraction)))
    rng = np.random.default_rng(seed + outer_fold_id)
    selected_val_event_idx = rng.choice(remaining_events.index.to_numpy(), size=min(n_val_events, len(remaining_events)), replace=False)
    val_events = remaining_events.loc[selected_val_event_idx].copy()
    val_patients = sorted(val_events["patient_id"].unique().tolist())

    val_neg_parts =[]
    for patient_id in val_patients:
        patient_excluded = set(val_events.loc[val_events["patient_id"] == patient_id, "global_interval"].tolist())
        val_neg_parts.append(
            _select_seizure_negative_pool(
                df,
                normal_only,
                normal_intervals,
                patient_id,
                exclude_intervals=patient_excluded,
            )
        )
    val_neg_pool = pd.concat(val_neg_parts, ignore_index=True) if val_neg_parts else df.iloc[0:0].copy()
    val_pos_pool = df.merge(
        val_events[["patient_id", "global_interval"]],
        on=["patient_id", "global_interval"],
        how="inner",
    )
    val_pos_pool = val_pos_pool.loc[val_pos_pool["class_label"] == 1].copy()

    excluded_row_ids = set(test_pos_pool["row_id"]).union(test_neg_pool["row_id"]).union(val_pos_pool["row_id"]).union(val_neg_pool["row_id"])
    train_pool = df.loc[~df["row_id"].isin(excluded_row_ids) & ~event_mask].copy().reset_index(drop=True)
    val_pool = pd.concat([val_pos_pool, val_neg_pool], ignore_index=True).drop_duplicates("row_id").reset_index(drop=True)
    test_pool = pd.concat([test_pos_pool, test_neg_pool], ignore_index=True).drop_duplicates("row_id").reset_index(drop=True)

    return FoldPools(
        train_pool=train_pool,
        val_pool=val_pool,
        test_pool=test_pool,
        metadata={
            "protocol": "seizure",
            "outer_fold_id": int(outer_fold_id),
            "test_group_id": held_out["group_id"],
            "test_patient": test_patient,
            "test_interval": test_interval,
        },
    )


def _select_seizure_negative_pool(
    df: pd.DataFrame,
    normal_only: pd.DataFrame,
    normal_intervals: pd.DataFrame,
    patient_id: str,
    exclude_intervals: set[int] | None = None,
) -> pd.DataFrame:
    exclude_intervals = exclude_intervals or set()
    recording_pool = df.loc[
        (df["patient_id"] == patient_id)
        & (df["class_label"] == 0)
        & (df["filename"].isin(normal_only.loc[normal_only["patient_id"] == patient_id, "filename"]))
        & (~df["global_interval"].isin(list(exclude_intervals)))
    ].copy()
    if not recording_pool.empty:
        return recording_pool

    interval_pool = df.loc[
        (df["patient_id"] == patient_id)
        & (df["class_label"] == 0)
        & (df["global_interval"].isin(normal_intervals.loc[normal_intervals["patient_id"] == patient_id, "global_interval"]))
        & (~df["global_interval"].isin(list(exclude_intervals)))
    ].copy()
    if not interval_pool.empty:
        return interval_pool

    return df.loc[
        (df["patient_id"] == patient_id)
        & (df["class_label"] == 0)
        & (~df["global_interval"].isin(list(exclude_intervals)))
    ].copy()


def _resolve_patient_fold(
    df: pd.DataFrame,
    assignments: pd.DataFrame,
    outer_fold_id: int,
    seed: int,
    val_fraction: float,
) -> FoldPools:
    held_out = assignments.loc[assignments["outer_fold_id"] == outer_fold_id].iloc[0]
    test_patient = held_out["patient_id"]
    test_pool = df.loc[df["patient_id"] == test_patient].copy()
    remaining_patients = assignments.loc[assignments["outer_fold_id"] != outer_fold_id, "patient_id"].tolist()
    n_val_patients = max(1, int(round(len(remaining_patients) * val_fraction)))
    rng = np.random.default_rng(seed + outer_fold_id)
    val_patients = sorted(rng.choice(np.array(remaining_patients), size=min(n_val_patients, len(remaining_patients)), replace=False).tolist())
    val_pool = df.loc[df["patient_id"].isin(val_patients)].copy().reset_index(drop=True)
    train_pool = df.loc[~df["patient_id"].isin(val_patients + [test_patient])].copy().reset_index(drop=True)
    return FoldPools(
        train_pool=train_pool,
        val_pool=val_pool,
        test_pool=test_pool.reset_index(drop=True),
        metadata={
            "protocol": "patient",
            "outer_fold_id": int(outer_fold_id),
            "test_group_id": test_patient,
            "test_patient": test_patient,
        },
    )
