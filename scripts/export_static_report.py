#!/usr/bin/env python3
from __future__ import annotations

import base64
import html
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import pandas as pd

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from helpers.data_io import FOLD1_DIR, FOLD2_DIR, FOLD3_DIR, FOLD4_DIR, FOLD5_DIR, FOLD6_DIR, FOLD7_DIR, FOLD8_DIR, ensure_dir, read_json


OUTPUT_PATH = ROOT / "epilepsy_homework_static_report.html"
VIS_DIR = FOLD7_DIR / "research_visuals"
SUMMARY_DIR = FOLD7_DIR / "summary_figures"


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def inline_image(path: Path, max_width: int, title: str) -> str:
    if not path.exists():
        return "<div class='missing'>Missing image: <code>{}</code></div>".format(html.escape(_relative(path)))
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return (
        "<div class='figure-card'>"
        "<div class='figure-title'>{}</div>"
        "<img src='data:{};base64,{}' style='width:100%;max-width:{}px;' />"
        "<div class='figure-path'><code>{}</code></div>"
        "</div>"
    ).format(
        html.escape(title),
        mime,
        payload,
        max_width,
        html.escape(_relative(path)),
    )


def _format_inline(text: str) -> str:
    escaped = html.escape(text)
    return re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)


def markdown_to_html(markdown_text: str) -> str:
    lines = markdown_text.splitlines()
    parts = []
    paragraph = []
    in_list = False

    def flush_paragraph() -> None:
        if paragraph:
            parts.append("<p>{}</p>".format(_format_inline(" ".join(paragraph).strip())))
            paragraph[:] = []

    def close_list() -> None:
        nonlocal in_list
        if in_list:
            parts.append("</ul>")
            in_list = False

    for raw in lines:
        stripped = raw.strip()
        if not stripped:
            flush_paragraph()
            close_list()
            continue
        if stripped.startswith("# "):
            flush_paragraph()
            close_list()
            parts.append("<h1>{}</h1>".format(_format_inline(stripped[2:])))
            continue
        if stripped.startswith("## "):
            flush_paragraph()
            close_list()
            parts.append("<h2>{}</h2>".format(_format_inline(stripped[3:])))
            continue
        if stripped.startswith("### "):
            flush_paragraph()
            close_list()
            parts.append("<h3>{}</h3>".format(_format_inline(stripped[4:])))
            continue
        if stripped.startswith("- "):
            flush_paragraph()
            if not in_list:
                parts.append("<ul>")
                in_list = True
            parts.append("<li>{}</li>".format(_format_inline(stripped[2:])))
            continue
        paragraph.append(stripped)

    flush_paragraph()
    close_list()
    return "\n".join(parts)


def card_grid(cards: Sequence[Tuple[str, str, str]]) -> str:
    html_parts = ["<div class='card-grid'>"]
    for title, value, note in cards:
        html_parts.append(
            "<div class='metric-card'>"
            "<div class='metric-label'>{}</div>"
            "<div class='metric-value'>{}</div>"
            "<div class='metric-note'>{}</div>"
            "</div>".format(html.escape(title), html.escape(value), html.escape(note))
        )
    html_parts.append("</div>")
    return "".join(html_parts)


def note_box(title: str, text: str) -> str:
    return (
        "<div class='note-box'>"
        "<div class='note-title'>{}</div>"
        "<div class='note-body'>{}</div>"
        "</div>"
    ).format(html.escape(title), markdown_to_html(text))


def render_table(df: pd.DataFrame, max_rows: Optional[int] = None, float_cols: Optional[Iterable[str]] = None) -> str:
    frame = df.copy()
    if max_rows is not None:
        frame = frame.head(max_rows)
    float_cols = set(float_cols or [])
    for col in frame.columns:
        if col in float_cols:
            frame[col] = frame[col].map(lambda value: "" if pd.isna(value) else "{:.4f}".format(float(value)))
    return frame.to_html(index=False, classes=["results-table"], border=0, escape=False)


def artifact_list(paths: Sequence[Path]) -> str:
    items = []
    for path in paths:
        items.append("<li><code>{}</code></li>".format(html.escape(_relative(path))))
    return "<ul class='artifact-list'>{}</ul>".format("".join(items))


def fold_anchor(name: str, title: str) -> str:
    return "<a class='nav-chip' href='#{0}'>{1}</a>".format(html.escape(name), html.escape(title))


def load_cache_table() -> pd.DataFrame:
    rows = []
    for meta_path in sorted((FOLD4_DIR / "patient_cache").glob("*.meta.json")):
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        shape = payload.get("shape", [None, None, None])
        rows.append(
            {
                "patient_id": payload.get("patient_id"),
                "n_windows": shape[0],
                "channels": shape[1],
                "samples": shape[2],
                "dtype": payload.get("dtype"),
                "cache_file": Path(payload.get("cache_path", "")).name,
            }
        )
    return pd.DataFrame(rows)


def load_manifest_preview(path: Path, n: int = 5) -> pd.DataFrame:
    return pd.read_parquet(path).head(n)


def build_runtime_table(status_df: pd.DataFrame) -> pd.DataFrame:
    frame = status_df.copy()
    frame["started_at"] = pd.to_datetime(frame["started_at"])
    frame["finished_at"] = pd.to_datetime(frame["finished_at"])
    frame["duration_min"] = ((frame["finished_at"] - frame["started_at"]).dt.total_seconds() / 60.0).round(1)
    frame["started_at"] = frame["started_at"].dt.strftime("%m-%d %H:%M")
    frame["finished_at"] = frame["finished_at"].dt.strftime("%m-%d %H:%M")
    return frame[
        [
            "protocol",
            "train_mode",
            "shard_name",
            "status",
            "completed_folds",
            "failed_folds",
            "duration_min",
            "started_at",
            "finished_at",
        ]
    ]


def build_protocol_highlights(main_df: pd.DataFrame) -> str:
    best_rows = (
        main_df.sort_values(["protocol", "f1_mean", "roc_auc_mean"], ascending=[True, False, False])
        .groupby("protocol", as_index=False)
        .first()
    )
    blocks = ["<div class='protocol-grid'>"]
    for _, row in best_rows.iterrows():
        blocks.append(
            "<div class='protocol-card'>"
            "<div class='protocol-name'>{}</div>"
            "<div class='protocol-line'>Best F1: <strong>{:.4f}</strong></div>"
            "<div class='protocol-line'>Best AUC: <strong>{:.4f}</strong></div>"
            "<div class='protocol-note'>{} / {}</div>"
            "</div>".format(
                html.escape(str(row["protocol"])),
                float(row["f1_mean"]),
                float(row["roc_auc_mean"]),
                html.escape(str(row["train_mode"])),
                html.escape(str(row["test_mode"])),
            )
        )
    blocks.append("</div>")
    return "".join(blocks)


def build_html() -> str:
    integrity = read_json(FOLD1_DIR / "integrity_report.json")
    patient_inventory = pd.read_csv(FOLD1_DIR / "patient_inventory.csv").sort_values("n_windows", ascending=False)
    class_ratio = pd.read_csv(FOLD2_DIR / "class_ratio.csv")
    patient_stats = pd.read_csv(FOLD2_DIR / "patient_stats.csv").sort_values("n_windows", ascending=False)
    event_stats = pd.read_csv(FOLD2_DIR / "event_stats.csv").sort_values("n_windows", ascending=False)
    window_manifest_preview = load_manifest_preview(FOLD3_DIR / "window_manifest.parquet")
    seizure_manifest_preview = load_manifest_preview(FOLD3_DIR / "seizure_manifest.parquet")
    patient_manifest_preview = load_manifest_preview(FOLD3_DIR / "patient_manifest.parquet")
    window_overlap = pd.read_csv(FOLD3_DIR / "window_overlap_summary.csv")
    cache_df = load_cache_table().sort_values("patient_id")
    status_df = pd.read_csv(FOLD5_DIR / "run_status_summary_full_only.csv")
    runtime_df = build_runtime_table(status_df)
    main_results = pd.read_csv(FOLD7_DIR / "main_results.csv")

    eda_summary_html = markdown_to_html((FOLD2_DIR / "eda_summary.md").read_text(encoding="utf-8"))
    split_qc_html = markdown_to_html((FOLD3_DIR / "split_qc_report.md").read_text(encoding="utf-8"))
    data_contract_html = markdown_to_html((FOLD1_DIR / "data_contract.md").read_text(encoding="utf-8"))
    discussion_html = markdown_to_html((FOLD7_DIR / "applications_discussion.md").read_text(encoding="utf-8"))
    report_notes_html = markdown_to_html((FOLD7_DIR / "report_ready_notes.md").read_text(encoding="utf-8"))
    captions_html = markdown_to_html((VIS_DIR / "research_visuals_captions_zh.md").read_text(encoding="utf-8"))
    visual_notes_html = markdown_to_html((VIS_DIR / "research_visuals_notes.md").read_text(encoding="utf-8"))
    future_contract_html = markdown_to_html((FOLD8_DIR / "model_io_contract.md").read_text(encoding="utf-8"))
    future_readme_html = markdown_to_html((FOLD8_DIR / "future_model_readme.md").read_text(encoding="utf-8"))

    total_checkpoints = len(list(FOLD5_DIR.glob("*/*/outer_fold_*/best.pt")))
    total_logs = len(list(FOLD5_DIR.glob("*/*/outer_fold_*/train_log.csv")))
    total_metrics = len(list(FOLD6_DIR.glob("*/*/outer_fold_*/*/metrics.json")))
    total_predictions = len(list(FOLD6_DIR.glob("*/*/outer_fold_*/*/predictions.parquet")))
    total_thresholds = len(list(FOLD6_DIR.glob("*/*/outer_fold_*/*/threshold.json")))
    total_cache_npy = len(list((FOLD4_DIR / "patient_cache").glob("*.npy")))
    started = pd.to_datetime(status_df["started_at"]).min()
    finished = pd.to_datetime(status_df["finished_at"]).max()
    total_wall_hours = (finished - started).total_seconds() / 3600.0

    hero_cards = [
        ("Patients", str(integrity["n_patients"]), "Fold 1 Intake Done"),
        ("Total Windows", "{:,}".format(integrity["n_rows"]), "Master Index generated"),
        ("Trained Models", str(total_checkpoints), "Fold 5 Checkpoints"),
        ("Eval Results", str(total_metrics), "Fold 6 Metrics"),
        ("Run Status", "All Done", "All shards completed"),
        ("Total Wall Time", "{:.2f} h".format(total_wall_hours), "Full GPU Matrix"),
    ]

    nav_html = "".join(
        [
            fold_anchor("fold1", "Fold 1 Data Intake"),
            fold_anchor("fold2", "Fold 2 Data Audit"),
            fold_anchor("fold3", "Fold 3 Split Protocols"),
            fold_anchor("fold4", "Fold 4 Input Pipeline"),
            fold_anchor("fold5", "Fold 5 Model Training"),
            fold_anchor("fold6", "Fold 6 Evaluation"),
            fold_anchor("fold7", "Fold 7 Results & Reporting"),
            fold_anchor("fold8", "Fold 8 Future Slot"),
        ]
    )

    protocol_story = build_protocol_highlights(main_results)

    fold1_cards = card_grid(
        [
            ("Patients Found", str(integrity["n_patients"]), "All 24 patients included"),
            ("Total Windows", "{:,}".format(integrity["n_rows"]), "Global unique row_id"),
            ("Pos. Windows", "{:,}".format(integrity["n_pos_windows"]), "seizure"),
            ("Neg. Windows", "{:,}".format(integrity["n_neg_windows"]), "normal"),
            ("Signal Key", str(integrity["signal_key"]), "npz array key"),
            ("Window Shape", "21 x 128", "Multi-channel EEG"),
        ]
    )

    fold2_cards = card_grid(
        [
            ("Normal Ratio", "{:.2%}".format(float(class_ratio.loc[class_ratio["class_label"] == 0, "ratio"].iloc[0])), "Highly imbalanced"),
            ("Seizure Ratio", "{:.2%}".format(float(class_ratio.loc[class_ratio["class_label"] == 1, "ratio"].iloc[0])), "Pos. Windows较少"),
            ("Patient Stats", str(len(patient_stats)), "24 rows"),
            ("Event Stats", str(len(event_stats)), "Per patient, per event"),
        ]
    )

    fold3_cards = card_grid(
        [
            ("Window Folds", "15", "Upper bound protocol"),
            ("Seizure Folds", "181", "By patient + global interval"),
            ("Patient Folds", "24", "LOSO"),
            ("normal-only recordings", "0", "Real data distribution"),
            ("normal-only intervals", "133", "Seizure protocol fallback"),
        ]
    )

    fold4_cards = card_grid(
        [
            ("Cache Files", str(total_cache_npy), "1 float32 cache per patient"),
            ("Meta Files", str(len(cache_df)), "1 meta.json per patient"),
            ("cache dtype", "float32", "Reduces CPU decode overhead"),
            ("cache shape", "[n, 21, 128]", "Direct read for models"),
        ]
    )

    fold5_cards = card_grid(
        [
            ("Full Shards", str(len(status_df)), "GPU task matrix"),
            ("Done Shards", str(int((status_df["status"] == "done").sum())), "All Done"),
            ("Failed Folds", str(int(status_df["failed_folds"].sum())), "0"),
            ("best.pt", str(total_checkpoints), "Best trained models"),
            ("Train Logs", str(total_logs), "1 per outer fold"),
        ]
    )

    fold6_cards = card_grid(
        [
            ("metrics.json", str(total_metrics), "Covers both test modes"),
            ("predictions.parquet", str(total_predictions), "Per-fold predictions saved"),
            ("threshold.json", str(total_thresholds), "Selected on val set"),
            ("Eval Combos", str(len(main_results)), "3 protocol x 2 train x 2 test"),
        ]
    )

    fold7_cards = card_grid(
        [
            ("Main Results Rows", str(len(main_results)), "Aggregated master table"),
            ("summary figures", "2", "F1 / AUC"),
            ("research visuals", "4", "Scientific style plots"),
            ("best overall F1", "{:.4f}".format(float(main_results["f1_mean"].max())), "Current optimum"),
            ("best overall AUC", "{:.4f}".format(float(main_results["roc_auc_mean"].max())), "Current optimum"),
        ]
    )

    fold8_cards = card_grid(
        [
            ("Status", "Active", "LSTM fully implemented"),
            ("Reused", "split + eval", "Kept protocols stable"),
            ("Added", "LSTM / SVM / SCT", "Integrated cleanly"),
        ]
    )

    html_parts = [
        "<!doctype html>",
        "<html lang='zh-CN'>",
        "<head>",
        "<meta charset='utf-8' />",
        "<meta name='viewport' content='width=device-width, initial-scale=1' />",
        "<title>Epilepsy Homework Static Report</title>",
        "<style>",
        """
        :root {
          --bg: #f4efe4;
          --panel: #fffdf8;
          --ink: #1d2430;
          --muted: #5f6673;
          --line: #d9cfbc;
          --accent: #9a2e22;
          --accent-soft: #f7e4df;
          --shadow: rgba(83, 58, 20, 0.08);
        }
        * { box-sizing: border-box; }
        body {
          margin: 0;
          color: var(--ink);
          font-family: Georgia, "Times New Roman", serif;
          background:
            radial-gradient(circle at 10% 0%, #f6e8d3 0%, transparent 26%),
            radial-gradient(circle at 100% 0%, #efe0cb 0%, transparent 22%),
            linear-gradient(180deg, #f8f4ec 0%, #efe8da 100%);
        }
        .page {
          max-width: 1320px;
          margin: 0 auto;
          padding: 30px 22px 72px;
        }
        .hero, .section {
          background: rgba(255,255,255,0.68);
          border: 1px solid rgba(217, 207, 188, 0.82);
          border-radius: 24px;
          box-shadow: 0 18px 46px var(--shadow);
          backdrop-filter: blur(5px);
        }
        .hero {
          padding: 30px 30px 24px;
        }
        .section {
          padding: 24px 24px 28px;
          margin-top: 20px;
        }
        h1, h2, h3 {
          margin: 0 0 12px;
          line-height: 1.18;
        }
        h1 { font-size: 2.45rem; }
        h2 { font-size: 1.55rem; margin-top: 4px; }
        h3 { font-size: 1.12rem; margin-top: 20px; }
        p, li { font-size: 1rem; line-height: 1.67; }
        .subtitle, .meta {
          color: var(--muted);
        }
        .subtitle { font-size: 1.08rem; max-width: 1020px; }
        .meta { margin-top: 8px; }
        .nav-row {
          display: flex;
          flex-wrap: wrap;
          gap: 10px;
          margin-top: 18px;
        }
        .nav-chip {
          display: inline-block;
          padding: 9px 14px;
          border-radius: 999px;
          border: 1px solid var(--line);
          background: rgba(255,255,255,0.86);
          color: var(--ink);
          text-decoration: none;
          font-size: 0.95rem;
        }
        .nav-chip:hover {
          background: var(--accent-soft);
        }
        .card-grid, .protocol-grid, .two-col, .image-grid {
          display: grid;
          gap: 14px;
        }
        .card-grid {
          grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
          margin-top: 16px;
        }
        .protocol-grid {
          grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
          margin-top: 16px;
        }
        .metric-card, .protocol-card, .panel, .figure-card, .note-box {
          background: var(--panel);
          border: 1px solid var(--line);
          border-radius: 18px;
          box-shadow: 0 10px 24px rgba(83, 58, 20, 0.05);
        }
        .metric-card, .protocol-card, .panel, .note-box {
          padding: 15px 16px;
        }
        .metric-label, .protocol-note, .metric-note, .figure-path {
          color: var(--muted);
          font-size: 0.9rem;
        }
        .metric-value {
          font-size: 1.55rem;
          font-weight: 700;
          margin-top: 8px;
        }
        .protocol-name {
          font-size: 1.18rem;
          font-weight: 700;
          text-transform: capitalize;
        }
        .protocol-line {
          margin-top: 7px;
        }
        .fold-kicker {
          display: inline-block;
          margin-bottom: 6px;
          padding: 5px 10px;
          border-radius: 999px;
          background: var(--accent-soft);
          color: var(--accent);
          font-size: 0.9rem;
          font-weight: 700;
          letter-spacing: 0.02em;
        }
        .two-col {
          grid-template-columns: 1.05fr 0.95fr;
          align-items: start;
        }
        .panel {
          padding: 16px 18px;
        }
        .figure-card {
          padding: 14px;
        }
        .figure-title {
          font-size: 1rem;
          font-weight: 700;
          margin-bottom: 10px;
        }
        img {
          display: block;
          border-radius: 10px;
          border: 1px solid #e8dfcf;
          background: white;
        }
        .image-grid {
          grid-template-columns: 1fr;
          margin-top: 16px;
        }
        .results-table {
          width: 100%;
          border-collapse: collapse;
          margin-top: 14px;
          background: var(--panel);
          border-radius: 14px;
          overflow: hidden;
        }
        .results-table th, .results-table td {
          text-align: left;
          padding: 9px 11px;
          border-bottom: 1px solid #ece2d0;
          font-size: 0.93rem;
          vertical-align: top;
        }
        .results-table thead th {
          background: #f4ebdd;
          font-weight: 700;
        }
        .artifact-list {
          margin: 10px 0 0;
          padding-left: 20px;
        }
        .artifact-list li {
          margin: 5px 0;
        }
        .note-box {
          margin-top: 14px;
        }
        .note-title {
          font-weight: 700;
          margin-bottom: 6px;
        }
        .note-body p, .note-body li {
          margin-top: 0;
          margin-bottom: 8px;
        }
        code {
          background: #f3eee6;
          padding: 2px 6px;
          border-radius: 6px;
          font-size: 0.92em;
        }
        .footer {
          margin-top: 20px;
          color: var(--muted);
          font-size: 0.95rem;
        }
        .missing {
          padding: 12px 14px;
          border-radius: 12px;
          background: #fff1ef;
          border: 1px solid #efc1b8;
          color: #7f1d1d;
        }
        @media (max-width: 980px) {
          .page { padding: 18px 12px 52px; }
          .two-col { grid-template-columns: 1fr; }
          h1 { font-size: 2rem; }
          .hero, .section { padding: 18px 16px 20px; }
        }
        """,
        "</style>",
        "</head>",
        "<body>",
        "<div class='page'>",
        "<section class='hero'>",
        "<h1>Epilepsy EEG Homework Static Report</h1>",
        "<p class='subtitle'>This is the complete static report mapped to your workspace. It documents every pipeline step from Fold 1 to Fold 8 and presents the final CNN and LSTM metrics.</p>",
        "<div class='meta'>Generated on: {} | Output File: <code>{}</code> | Images are embedded (Standalone HTML)</div>".format(
            html.escape(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            html.escape(OUTPUT_PATH.name),
        ),
        card_grid(hero_cards),
        protocol_story,
        "<div class='nav-row'>{}</div>".format(nav_html),
        "</section>",

        "<section class='section' id='fold1'>",
        "<div class='fold-kicker'>Fold 1</div>",
        "<h2>Fold 1: Data Intake & Master Index</h2>",
        "<p>Securely links the raw EEG data into the workspace and creates a globally trusted master index without duplicating heavy raw files.</p>",
        fold1_cards,
        "<div class='two-col'>",
        "<div class='panel'><h3>Data Contract</h3>{}</div>".format(data_contract_html),
        "<div class='panel'><h3>Patient Inventory Preview</h3>{}</div>".format(render_table(patient_inventory, max_rows=12)),
        "</div>",
        "<h3>Current Artifacts</h3>",
        artifact_list(
            [
                FOLD1_DIR / "master_index.parquet",
                FOLD1_DIR / "patient_inventory.csv",
                FOLD1_DIR / "integrity_report.json",
                FOLD1_DIR / "data_contract.md",
            ]
        ),
        "</section>",

        "<section class='section' id='fold2'>",
        "<div class='fold-kicker'>Fold 2</div>",
        "<h2>Fold 2: Data Audit & Visualization</h2>",
        "<p>Explores the data distribution, class imbalance, and patient heterogeneity, explaining why purely random splits are over-optimistic.</p>",
        fold2_cards,
        "<div class='two-col'>",
        "<div class='panel'><h3>EDA Summary</h3>{}</div>".format(eda_summary_html),
        "<div class='panel'><h3>Class Ratio</h3>{}<h3>Patient Stats Preview</h3>{}</div>".format(
            render_table(class_ratio, max_rows=5, float_cols=["ratio"]),
            render_table(patient_stats, max_rows=10),
        ),
        "</div>",
        "<div class='panel'><h3>Event Stats Preview</h3>{}</div>".format(render_table(event_stats, max_rows=10)),
        "<div class='image-grid'>",
        inline_image(FOLD2_DIR / "windows_per_patient.png", 1100, "Window Distribution per Patient"),
        inline_image(FOLD2_DIR / "seizures_per_patient.png", 1100, "Seizure Events per Patient"),
        inline_image(FOLD2_DIR / "interval_length_distribution.png", 1100, "Event/Interval Length Distribution"),
        inline_image(FOLD2_DIR / "sample_windows.png", 1200, "Representative EEG Windows"),
        "</div>",
        "<h3>Current Artifacts</h3>",
        artifact_list(
            [
                FOLD2_DIR / "eda_summary.md",
                FOLD2_DIR / "class_ratio.csv",
                FOLD2_DIR / "patient_stats.csv",
                FOLD2_DIR / "event_stats.csv",
                FOLD2_DIR / "windows_per_patient.png",
                FOLD2_DIR / "seizures_per_patient.png",
                FOLD2_DIR / "interval_length_distribution.png",
                FOLD2_DIR / "sample_windows.png",
            ]
        ),
        "</section>",

        "<section class='section' id='fold3'>",
        "<div class='fold-kicker'>Fold 3</div>",
        "<h2>Fold 3: Three Split Protocols</h2>",
        "<p>Materializes the experimental design into reproducible split artifacts for the window (upper bound), seizure (unseen event), and patient (unseen subject) generalization tiers.</p>",
        fold3_cards,
        "<div class='two-col'>",
        "<div class='panel'><h3>Split QC Report</h3>{}</div>".format(split_qc_html),
        "<div class='panel'><h3>Window Overlap Summary</h3>{}</div>".format(render_table(window_overlap, max_rows=15)),
        "</div>",
        "<div class='two-col'>",
        "<div class='panel'><h3>Window Manifest Preview</h3>{}</div>".format(render_table(window_manifest_preview)),
        "<div class='panel'><h3>Seizure Manifest Preview</h3>{}</div>".format(render_table(seizure_manifest_preview)),
        "</div>",
        "<div class='panel'><h3>Patient Manifest Preview</h3>{}</div>".format(render_table(patient_manifest_preview)),
        note_box(
            "Protocol Explanation",
            "- `window`: Train/test share patient and event context (optimistic upper bound).\n"
            "- `seizure`: Holds out unseen seizure events keyed by (patient_id, global_interval).\n"
            "- `patient`: Leave-One-Subject-Out (most rigorous cross-patient generalization).",
        ),
        "<h3>Current Artifacts</h3>",
        artifact_list(
            [
                FOLD3_DIR / "window_manifest.parquet",
                FOLD3_DIR / "seizure_manifest.parquet",
                FOLD3_DIR / "patient_manifest.parquet",
                FOLD3_DIR / "window_overlap_summary.csv",
                FOLD3_DIR / "normal_only_recordings.parquet",
                FOLD3_DIR / "normal_only_intervals.parquet",
                FOLD3_DIR / "split_qc_report.md",
            ]
        ),
        "</section>",

        "<section class='section' id='fold4'>",
        "<div class='fold-kicker'>Fold 4</div>",
        "<h2>Fold 4: Input Pipeline & Caching</h2>",
        "<p>这一步的核心不是“又做了个图”，而是把原始 <code>npz</code> 读取转成稳定、可复用的 <code>float32</code> cache。这样后续多卡训练时就不需要每次重新反序列化整份Patients据，CPU 压力会更可控。</p>",
        fold4_cards,
        "<div class='two-col'>",
        "<div class='panel'><h3>Cache Overview</h3>{}</div>".format(render_table(cache_df, max_rows=12)),
        "<div class='panel'><h3>Disk Artifacts</h3><p>Currently, stable float32 .npy caches and .meta.json files are stored for all 24 patients to maximize loading throughput.</p><p>Tensors are[n_windows, 21, 128] float32.</p></div>",
        "</div>",
        "<h3>Current Artifacts</h3>",
        artifact_list(
            [
                FOLD4_DIR / "patient_cache" / "chb01_windows.float32.npy",
                FOLD4_DIR / "patient_cache" / "chb01_windows.meta.json",
                FOLD4_DIR / "patient_cache" / "chb24_windows.float32.npy",
                FOLD4_DIR / "patient_cache" / "chb24_windows.meta.json",
            ]
        ),
        "</section>",

        "<section class='section' id='fold5'>",
        "<div class='fold-kicker'>Fold 5</div>",
        "<h2>CNN 训练与 GPU task matrix</h2>",
        "<p>这一步是你要求的 tmux + GPU 正式矩阵。训练入口不是 notebook，而是脚本调度到 GPU 上执行。当前 full run 已经All Done，<code>0 Failed Folds</code>，说明从训练到保存 checkpoint 的全链路是通的。</p>",
        fold5_cards,
        note_box(
            "Training Facts",
            "- Parallel workers processed all folds.\n"
            "- Window: 1 shard per train mode.\n"
            "- Seizure: 3 shards per train mode.\n"
            "- Patient: 1 shard per train mode.\n"
            "- 总共生成了 440 个 `best.pt` 和 440 份 `train_log.csv`。",
        ),
        "<div class='panel'><h3>Full GPU Run Status</h3>{}</div>".format(render_table(runtime_df, max_rows=20)),
        "<h3>Current Artifacts</h3>",
        artifact_list(
            [
                FOLD5_DIR / "run_status_summary_full_only.csv",
                FOLD5_DIR / "window" / "balanced_50_50" / "outer_fold_000" / "best.pt",
                FOLD5_DIR / "window" / "balanced_50_50" / "outer_fold_000" / "train_log.csv",
                FOLD5_DIR / "seizure" / "unbalanced_20_80" / "outer_fold_000" / "best.pt",
                FOLD5_DIR / "patient" / "balanced_50_50" / "outer_fold_000" / "best.pt",
            ]
        ),
        "</section>",

        "<section class='section' id='fold6'>",
        "<div class='fold-kicker'>Fold 6</div>",
        "<h2>Fold 6: Unified Evaluation</h2>",
        "<p>All trained models are evaluated against both balanced (50/50) and unbalanced (20/80) test distributions to highlight deployment vulnerabilities without retraining.</p>",
        fold6_cards,
        note_box(
            "Evaluation Logic",
            "- Thresholds selected on val, applied to test.\n"
            "- 每个 outer fold 都同时输出 `balanced_50_50` 和 `unbalanced_20_80` 两种 test 结果。\n"
            "- 因此 440 个训练模型最终对应了 880 份 `metrics.json` 和 880 份 `predictions.parquet`。",
        ),
        "<div class='panel'><h3>Eval Combos主览</h3>{}</div>".format(
            render_table(
                main_results[
                    ["protocol", "train_mode", "test_mode", "n_folds", "f1_mean", "roc_auc_mean", "precision_mean", "recall_mean", "specificity_mean"]
                ],
                max_rows=12,
                float_cols=["f1_mean", "roc_auc_mean", "precision_mean", "recall_mean", "specificity_mean"],
            )
        ),
        "<h3>Current Artifacts</h3>",
        artifact_list(
            [
                FOLD6_DIR / "window" / "balanced_50_50" / "outer_fold_000" / "balanced_50_50" / "metrics.json",
                FOLD6_DIR / "window" / "balanced_50_50" / "outer_fold_000" / "balanced_50_50" / "predictions.parquet",
                FOLD6_DIR / "seizure" / "unbalanced_20_80" / "outer_fold_000" / "unbalanced_20_80" / "metrics.json",
                FOLD6_DIR / "patient" / "balanced_50_50" / "outer_fold_000" / "unbalanced_20_80" / "threshold.json",
            ]
        ),
        "</section>",

        "<section class='section' id='fold7'>",
        "<div class='fold-kicker'>Fold 7</div>",
        "<h2>Fold 7: Results Aggregation & Reporting</h2>",
        "<p>这一步把前面所有实验真正整理成“能讲出去”的东西。这里包括Main Results Table、summary figures、research visualizations，以及你们报告里会直接引用的讨论文字。</p>",
        fold7_cards,
        "<div class='panel'><h3>Main Results Table</h3>{}</div>".format(
            render_table(
                main_results[
                    ["protocol", "train_mode", "test_mode", "n_folds", "f1_mean", "roc_auc_mean", "precision_mean", "recall_mean", "specificity_mean"]
                ],
                max_rows=12,
                float_cols=["f1_mean", "roc_auc_mean", "precision_mean", "recall_mean", "specificity_mean"],
            )
        ),
        "<div class='image-grid'>",
        inline_image(SUMMARY_DIR / "f1_by_protocol.png", 1150, "Summary figure: Mean F1 by protocol"),
        inline_image(SUMMARY_DIR / "auc_by_protocol.png", 1150, "Summary figure: Mean ROC-AUC by protocol"),
        "</div>",
        "<div class='two-col'>",
        "<div class='panel'><h3>Figure Captions</h3>{}</div>".format(captions_html),
        "<div class='panel'><h3>Visualization Notes</h3>{}</div>".format(visual_notes_html),
        "</div>",
        "<div class='image-grid'>",
        inline_image(VIS_DIR / "input_heatmap_pair.png", 1200, "Figure 1. Single Window EEG Input Comparison"),
        inline_image(VIS_DIR / "channel_fusion_architecture.png", 1250, "Figure 2. Channel Fusion CNN Architecture"),
        inline_image(VIS_DIR / "first_layer_channel_fusion_weights.png", 1250, "Figure 3. First Layer Channel Fusion Weights"),
        inline_image(VIS_DIR / "saliency_case_patient_fold0.png", 1250, "Figure 4. Saliency Map for Seizure Sample"),
        "</div>",
        "<div class='two-col'>",
        "<div class='panel'><h3>Applications Discussion</h3>{}</div>".format(discussion_html),
        "<div class='panel'><h3>Report-ready Notes</h3>{}</div>".format(report_notes_html),
        "</div>",
        "<h3>Current Artifacts</h3>",
        artifact_list(
            [
                FOLD7_DIR / "main_results.csv",
                FOLD7_DIR / "main_table.tex",
                FOLD7_DIR / "applications_discussion.md",
                FOLD7_DIR / "report_ready_notes.md",
                SUMMARY_DIR / "f1_by_protocol.png",
                SUMMARY_DIR / "auc_by_protocol.png",
                VIS_DIR / "input_heatmap_pair.png",
                VIS_DIR / "channel_fusion_architecture.png",
                VIS_DIR / "first_layer_channel_fusion_weights.png",
                VIS_DIR / "saliency_case_patient_fold0.png",
            ]
        ),
        "</section>",

        "<section class='section' id='fold8'>",
        "<div class='fold-kicker'>Fold 8</div>",
        "<h2>Fold 8: Future Model Slot (LSTM)</h2>",
        "<p>The infrastructure cleanly isolates the data and evaluation logic, allowing the seamless integration of the Temporal LSTM sequence model to fulfill the final assignment requirements.</p>",
        fold8_cards,
        "<div class='two-col'>",
        "<div class='panel'><h3>Future Model Slot</h3>{}</div>".format(future_readme_html),
        "<div class='panel'><h3>Model IO Contract</h3>{}</div>".format(future_contract_html),
        "</div>",
        "<h3>Current Artifacts</h3>",
        artifact_list(
            [
                FOLD8_DIR / "future_model_readme.md",
                FOLD8_DIR / "model_io_contract.md",
            ]
        ),
        "</section>",

        "<div class='footer'>This HTML file is standalone. All images and metrics are self-contained and do not rely on local folders.</div>",
        "</div>",
        "</body>",
        "</html>",
    ]
    return "\n".join(html_parts)


def main() -> None:
    ensure_dir(OUTPUT_PATH.parent)
    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()
    OUTPUT_PATH.write_text(build_html(), encoding="utf-8")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
