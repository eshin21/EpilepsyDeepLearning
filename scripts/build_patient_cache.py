#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from helpers.data_io import CACHE_DIR, build_master_index, ensure_patient_caches, load_master_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize patient float32 caches sequentially.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--patients", nargs="*")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_master_index(force=False)
    master = load_master_index()
    patient_ids = sorted(master["patient_id"].unique().tolist())
    if args.patients:
        allowed = set(args.patients)
        patient_ids = [patient_id for patient_id in patient_ids if patient_id in allowed]
    print(f"[build_patient_cache] cache_dir={CACHE_DIR}")
    print(f"[build_patient_cache] patients={patient_ids}")
    ensure_patient_caches(master, cache_dir=CACHE_DIR, patient_ids=patient_ids, force=args.force)
    print("[build_patient_cache] done")


if __name__ == "__main__":
    main()
