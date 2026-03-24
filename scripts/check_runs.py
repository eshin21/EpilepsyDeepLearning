#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from helpers.data_io import FOLD5_DIR, ensure_dir


def collect_status_rows(project_root: Path = ROOT) -> pd.DataFrame:
    runs_root = project_root / "fold5_cnn_training" / "runs"
    rows: list[dict[str, object]] = []
    for status_path in sorted(runs_root.glob("*/*/shard_*/status.json")):
        payload = json.loads(status_path.read_text(encoding="utf-8"))
        smoke = bool(payload.get("smoke", False))
        rows.append(
            {
                "protocol": payload.get("protocol"),
                "train_mode": payload.get("train_mode"),
                "shard_name": status_path.parent.name,
                "smoke": smoke,
                "status": payload.get("status"),
                "device": payload.get("device"),
                "completed_folds": len(payload.get("completed_folds", [])),
                "failed_folds": len(payload.get("failed_folds", [])),
                "started_at": payload.get("started_at"),
                "finished_at": payload.get("finished_at"),
                "test_modes": ",".join(payload.get("test_modes", [])),
                "status_path": str(status_path),
            }
        )
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    return summary.sort_values(
        by=["smoke", "protocol", "train_mode", "shard_name"],
        ascending=[True, True, True, True],
        ignore_index=True,
    )


def main() -> None:
    summary = collect_status_rows(ROOT)
    ensure_dir(FOLD5_DIR)
    output_path = FOLD5_DIR / "run_status_summary.csv"
    full_only_output_path = FOLD5_DIR / "run_status_summary_full_only.csv"
    summary.to_csv(output_path, index=False)
    full_only = summary.loc[~summary["smoke"]] if not summary.empty else summary
    full_only.to_csv(full_only_output_path, index=False)
    if summary.empty:
        print("No status.json files found yet.")
    else:
        print(summary.to_string(index=False))
        print(f"\nSaved status summary to {output_path}")
        print(f"Saved full-run-only summary to {full_only_output_path}")


if __name__ == "__main__":
    main()
