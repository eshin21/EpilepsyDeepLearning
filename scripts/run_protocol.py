#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from helpers.data_io import (
    CACHE_DIR,
    FOLD4_DIR,
    FOLD5_DIR,
    FOLD6_DIR,
    build_audit_artifacts,
    build_master_index,
    ensure_dir,
    ensure_patient_caches,
    load_master_index,
    write_json,
    write_yaml,
)
from helpers.eval import build_prediction_frame, compute_binary_metrics, save_evaluation_bundle
from helpers.model import load_trained_model, predict_with_model, train_model
from helpers.splits import (
    TEST_MODE_TO_RATIO,
    TRAIN_MODE_TO_RATIO,
    build_split_artifacts,
    list_outer_fold_ids,
    resolve_fold_pools,
    sample_rows_for_ratio,
    smoke_subset,
)
from scripts.aggregate_results import aggregate_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one protocol/train-mode slice of the epilepsy pipeline.")
    parser.add_argument("--protocol", choices=["window", "seizure", "patient"], required=True)
    parser.add_argument("--train-mode", choices=sorted(TRAIN_MODE_TO_RATIO), required=True)
    parser.add_argument("--test-modes", nargs="+", choices=sorted(TEST_MODE_TO_RATIO), default=sorted(TEST_MODE_TO_RATIO))
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--outer-fold-ids", nargs="*", type=int)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device, but torch.cuda.is_available() is False.")
    return device_arg


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_run_dir(protocol: str, train_mode: str, shard_index: int, num_shards: int, smoke: bool) -> Path:
    suffix = "_smoke" if smoke else ""
    return FOLD5_DIR / "runs" / protocol / train_mode / f"shard_{shard_index:02d}_of_{num_shards:02d}{suffix}"


def save_status(status_path: Path, payload: dict[str, object]) -> None:
    write_json(status_path, payload)


def plot_learning_curve(log_path: Path, output_path: Path) -> None:
    history = pd.read_csv(log_path)
    if history.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["epoch"], history["train_loss"], label="train_loss", color="#2b7fff")
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[1].plot(history["epoch"], history["val_auc"], label="val_auc", color="#d62728")
    axes[1].set_title("Validation AUC")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUC")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def fold_label(outer_fold_id: int) -> str:
    return f"outer_fold_{outer_fold_id:03d}"


def filter_fold_ids(all_fold_ids: list[int], args: argparse.Namespace) -> list[int]:
    fold_ids = all_fold_ids
    if args.outer_fold_ids:
        requested = set(args.outer_fold_ids)
        fold_ids = [fold_id for fold_id in fold_ids if fold_id in requested]
    fold_ids = [fold_id for fold_id in fold_ids if fold_id % args.num_shards == args.shard_index]
    if args.max_folds is not None:
        fold_ids = fold_ids[: args.max_folds]
    if args.smoke:
        fold_ids = fold_ids[:1]
    return fold_ids


def maybe_smoke(rows: pd.DataFrame, seed: int, enabled: bool, max_positive: int = 128, max_negative: int = 128) -> pd.DataFrame:
    if not enabled:
        return rows.reset_index(drop=True)
    return smoke_subset(rows, seed=seed, max_positive=max_positive, max_negative=max_negative)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    ensure_dir(ROOT / ".mplconfig")
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

    run_dir = make_run_dir(args.protocol, args.train_mode, args.shard_index, args.num_shards, args.smoke)
    ensure_dir(run_dir)
    status_path = run_dir / "status.json"
    run_config_path = run_dir / "run_config.yaml"

    initial_status = {
        "protocol": args.protocol,
        "train_mode": args.train_mode,
        "test_modes": args.test_modes,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "device": device,
        "smoke": args.smoke,
        "status": "running",
        "started_at": utcnow(),
        "completed_folds": [],
        "failed_folds": [],
        "errors": [],
    }
    save_status(status_path, initial_status)
    run_config = vars(args).copy()
    run_config["device"] = device
    write_yaml(run_config_path, run_config)
    print(f"[run_protocol] protocol={args.protocol} train_mode={args.train_mode} device={device} smoke={args.smoke}", flush=True)

    try:
        print("[run_protocol] building master index / audit / split artifacts", flush=True)
        build_master_index(force=False)
        master_index = load_master_index()
        build_audit_artifacts(master_index, force=False)
        build_split_artifacts(master_index, force=False)

        all_fold_ids = list_outer_fold_ids(master_index, args.protocol)
        selected_fold_ids = filter_fold_ids(all_fold_ids, args)
        if not selected_fold_ids:
            raise RuntimeError("No outer folds selected for this task.")

        initial_status["selected_fold_ids"] = selected_fold_ids
        save_status(status_path, initial_status)
        print(f"[run_protocol] selected_folds={selected_fold_ids}", flush=True)

        for outer_fold_id in selected_fold_ids:
            fold_name = fold_label(outer_fold_id)
            try:
                print(f"[run_protocol] resolving pools for {fold_name}", flush=True)
                pools = resolve_fold_pools(master_index, args.protocol, outer_fold_id, seed=args.seed)
                train_rows = sample_rows_for_ratio(
                    pools.train_pool,
                    target_positive_ratio=TRAIN_MODE_TO_RATIO[args.train_mode],
                    seed=args.seed + outer_fold_id,
                )
                val_rows = pools.val_pool.reset_index(drop=True)
                train_rows = maybe_smoke(train_rows, seed=args.seed + outer_fold_id, enabled=args.smoke, max_positive=256, max_negative=256)
                val_rows = maybe_smoke(val_rows, seed=args.seed + outer_fold_id + 1, enabled=args.smoke, max_positive=64, max_negative=64)
                print(
                    f"[run_protocol] {fold_name} rows train={len(train_rows)} val={len(val_rows)} test_pool={len(pools.test_pool)}",
                    flush=True,
                )

                patient_ids_needed = sorted(
                    set(train_rows["patient_id"].unique()).union(val_rows["patient_id"].unique()).union(pools.test_pool["patient_id"].unique())
                )
                print(f"[run_protocol] materializing caches for {len(patient_ids_needed)} patients", flush=True)
                ensure_patient_caches(master_index, cache_dir=CACHE_DIR, patient_ids=patient_ids_needed, force=False)

                fold4_dir = ensure_dir(FOLD4_DIR / args.protocol / args.train_mode / fold_name)
                fold5_dir = ensure_dir(FOLD5_DIR / args.protocol / args.train_mode / fold_name)
                fold6_dir = ensure_dir(FOLD6_DIR / args.protocol / args.train_mode / fold_name)

                print(f"[run_protocol] training {fold_name}", flush=True)
                training = train_model(
                    train_rows=train_rows,
                    val_rows=val_rows,
                    output_dir=fold5_dir,
                    device=device,
                    seed=args.seed + outer_fold_id,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    max_epochs=args.max_epochs,
                    patience=args.patience,
                    force=args.force,
                )
                norm_stats_copy = fold4_dir / "norm_stats.json"
                norm_stats_copy.write_text(training.norm_stats_path.read_text(encoding="utf-8"), encoding="utf-8")
                plot_learning_curve(training.log_path, fold5_dir / "learning_curve.png")

                model, mean, std, threshold = load_trained_model(training.checkpoint_path, device=device)
                loader_smoke = {
                    "protocol": args.protocol,
                    "train_mode": args.train_mode,
                    "outer_fold_id": outer_fold_id,
                    "device": device,
                    "train_rows": int(len(train_rows)),
                    "val_rows": int(len(val_rows)),
                }
                write_json(fold4_dir / "loader_smoke_report.json", loader_smoke)

                for test_mode in args.test_modes:
                    print(f"[run_protocol] evaluating {fold_name} test_mode={test_mode}", flush=True)
                    test_rows = sample_rows_for_ratio(
                        pools.test_pool,
                        target_positive_ratio=TEST_MODE_TO_RATIO[test_mode],
                        seed=args.seed + outer_fold_id + 1000,
                    )
                    test_rows = maybe_smoke(test_rows, seed=args.seed + outer_fold_id + 2, enabled=args.smoke, max_positive=64, max_negative=64)
                    raw_predictions = predict_with_model(
                        model=model,
                        rows=test_rows,
                        mean=mean,
                        std=std,
                        device=torch.device(device),
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                    )
                    prediction_frame = build_prediction_frame(
                        prediction_df=raw_predictions,
                        threshold=threshold,
                        protocol=args.protocol,
                        outer_fold_id=outer_fold_id,
                        train_mode=args.train_mode,
                        test_mode=test_mode,
                        checkpoint_path=training.checkpoint_path,
                    )
                    metrics = compute_binary_metrics(
                        prediction_frame["y_true"].to_numpy(),
                        prediction_frame["y_score"].to_numpy(),
                        threshold=threshold,
                    )
                    metrics.update(
                        {
                            "protocol": args.protocol,
                            "outer_fold_id": outer_fold_id,
                            "train_mode": args.train_mode,
                            "test_mode": test_mode,
                            "test_group_id": pools.metadata["test_group_id"],
                            "n_train_rows": int(len(train_rows)),
                            "n_val_rows": int(len(val_rows)),
                            "n_test_rows": int(len(test_rows)),
                        }
                    )
                    save_evaluation_bundle(fold6_dir / test_mode, prediction_frame, metrics)

                initial_status["completed_folds"].append(outer_fold_id)
                save_status(status_path, initial_status)
                print(f"[run_protocol] completed {fold_name}", flush=True)
            except Exception as fold_error:  # noqa: PERF203
                error_blob = {
                    "outer_fold_id": outer_fold_id,
                    "error": str(fold_error),
                    "traceback": traceback.format_exc(),
                }
                initial_status["failed_folds"].append(outer_fold_id)
                initial_status["errors"].append(error_blob)
                save_status(status_path, initial_status)
                print(f"[run_protocol] failed {fold_name}: {fold_error}", flush=True)
                if not args.keep_going:
                    raise

        print("[run_protocol] aggregating results", flush=True)
        aggregate_results(ROOT)
        initial_status["status"] = "done" if not initial_status["failed_folds"] else "partial"
        initial_status["finished_at"] = utcnow()
        save_status(status_path, initial_status)
        print(f"[run_protocol] finished with status={initial_status['status']}", flush=True)
    except Exception as error:
        initial_status["status"] = "failed"
        initial_status["finished_at"] = utcnow()
        initial_status["errors"].append({"error": str(error), "traceback": traceback.format_exc()})
        save_status(status_path, initial_status)
        print(f"[run_protocol] fatal error: {error}", flush=True)
        raise


if __name__ == "__main__":
    main()
