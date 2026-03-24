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
            "<div class='protocol-line'>最佳 F1: <strong>{:.4f}</strong></div>"
            "<div class='protocol-line'>最佳 AUC: <strong>{:.4f}</strong></div>"
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
        ("病人数", str(integrity["n_patients"]), "fold1 数据接入完成"),
        ("窗口总数", "{:,}".format(integrity["n_rows"]), "master_index 全量索引"),
        ("训练模型数", str(total_checkpoints), "fold5 最优 checkpoint"),
        ("评估结果数", str(total_metrics), "fold6 metrics.json"),
        ("运行状态", "全部完成", "10/10 full shard done"),
        ("总墙钟时间", "{:.2f} h".format(total_wall_hours), "2026-03-19 full GPU matrix"),
    ]

    nav_html = "".join(
        [
            fold_anchor("fold1", "Fold 1 数据接入"),
            fold_anchor("fold2", "Fold 2 数据分析"),
            fold_anchor("fold3", "Fold 3 数据切分"),
            fold_anchor("fold4", "Fold 4 输入管线"),
            fold_anchor("fold5", "Fold 5 CNN 训练"),
            fold_anchor("fold6", "Fold 6 评估"),
            fold_anchor("fold7", "Fold 7 结果汇总"),
            fold_anchor("fold8", "Fold 8 扩展位"),
        ]
    )

    protocol_story = build_protocol_highlights(main_results)

    fold1_cards = card_grid(
        [
            ("发现病人", str(integrity["n_patients"]), "全部 24 个病人已纳入"),
            ("窗口总数", "{:,}".format(integrity["n_rows"]), "全局唯一 row_id"),
            ("正类窗口", "{:,}".format(integrity["n_pos_windows"]), "seizure"),
            ("负类窗口", "{:,}".format(integrity["n_neg_windows"]), "normal"),
            ("信号键", str(integrity["signal_key"]), "npz 数组键"),
            ("窗口形状", "21 x 128", "多通道 EEG 单窗口"),
        ]
    )

    fold2_cards = card_grid(
        [
            ("normal 占比", "{:.2%}".format(float(class_ratio.loc[class_ratio["class_label"] == 0, "ratio"].iloc[0])), "类别不平衡明显"),
            ("seizure 占比", "{:.2%}".format(float(class_ratio.loc[class_ratio["class_label"] == 1, "ratio"].iloc[0])), "正类窗口较少"),
            ("病人统计表", str(len(patient_stats)), "24 行"),
            ("事件统计表", str(len(event_stats)), "逐病人逐事件"),
        ]
    )

    fold3_cards = card_grid(
        [
            ("window folds", "15", "上界协议"),
            ("seizure folds", "181", "按 patient + global_interval"),
            ("patient folds", "24", "LOSO"),
            ("normal-only recordings", "0", "数据真实情况"),
            ("normal-only intervals", "133", "seizure 协议回退来源"),
        ]
    )

    fold4_cards = card_grid(
        [
            ("cache 文件", str(total_cache_npy), "每个病人 1 个 float32 cache"),
            ("meta 文件", str(len(cache_df)), "每个病人 1 个 meta.json"),
            ("cache dtype", "float32", "减少 CPU 解码开销"),
            ("cache shape", "[n, 21, 128]", "直接给 CNN 读"),
        ]
    )

    fold5_cards = card_grid(
        [
            ("full shards", str(len(status_df)), "GPU 任务矩阵"),
            ("done shards", str(int((status_df["status"] == "done").sum())), "全部完成"),
            ("failed folds", str(int(status_df["failed_folds"].sum())), "0"),
            ("best.pt", str(total_checkpoints), "训练最优模型"),
            ("train logs", str(total_logs), "每个 outer fold 一份"),
        ]
    )

    fold6_cards = card_grid(
        [
            ("metrics.json", str(total_metrics), "2 个 test mode 全覆盖"),
            ("predictions.parquet", str(total_predictions), "逐 fold 预测已落盘"),
            ("threshold.json", str(total_thresholds), "val 上选阈值"),
            ("评估组合", str(len(main_results)), "3 protocol x 2 train x 2 test"),
        ]
    )

    fold7_cards = card_grid(
        [
            ("main results 行数", str(len(main_results)), "聚合主表"),
            ("summary figures", "2", "F1 / AUC"),
            ("research visuals", "4", "科研风解释图"),
            ("best overall F1", "{:.4f}".format(float(main_results["f1_mean"].max())), "当前 CNN 最优"),
            ("best overall AUC", "{:.4f}".format(float(main_results["roc_auc_mean"].max())), "当前 CNN 最优"),
        ]
    )

    fold8_cards = card_grid(
        [
            ("当前状态", "预留", "V1 不实现第二模型"),
            ("会复用", "split + eval", "不改前面协议"),
            ("不在本轮", "LSTM / SVM / SCT", "只保留接口"),
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
        "<p class='subtitle'>这不是刚才那份“结果浓缩版”，而是按你当前真实工程目录 <code>fold1</code> 到 <code>fold8</code> 展开的完整版静态报告。它的目标是让同学只拿到一个 HTML 文件，也能清楚看到每一步到底做了什么、现在做到哪了、产出了哪些证据。</p>",
        "<div class='meta'>生成时间：{} | 输出文件：<code>{}</code> | 本文件已内嵌图片，可单独分享</div>".format(
            html.escape(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            html.escape(OUTPUT_PATH.name),
        ),
        card_grid(hero_cards),
        protocol_story,
        "<div class='nav-row'>{}</div>".format(nav_html),
        "</section>",

        "<section class='section' id='fold1'>",
        "<div class='fold-kicker'>Fold 1</div>",
        "<h2>数据接入与总索引</h2>",
        "<p>这一步负责把服务器上的原始癫痫数据安全接入到作业目录，并建立唯一可信的总索引。当前工程没有复制原始大文件，而是通过软链接读取，并在本地生成 <code>master_index.parquet</code>、完整性报告和病人清单。</p>",
        fold1_cards,
        "<div class='two-col'>",
        "<div class='panel'><h3>数据契约</h3>{}</div>".format(data_contract_html),
        "<div class='panel'><h3>病人清单预览</h3>{}</div>".format(render_table(patient_inventory, max_rows=12)),
        "</div>",
        "<h3>当前产物</h3>",
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
        "<h2>数据分析与可视化审计</h2>",
        "<p>这一步把数据本身长什么样、偏不偏、病人之间差异大不大讲清楚。它直接服务于你们报告里的“数据描述”和“为什么 window holdout 会更乐观”这部分。</p>",
        fold2_cards,
        "<div class='two-col'>",
        "<div class='panel'><h3>EDA Summary</h3>{}</div>".format(eda_summary_html),
        "<div class='panel'><h3>类别比例</h3>{}<h3>病人统计预览</h3>{}</div>".format(
            render_table(class_ratio, max_rows=5, float_cols=["ratio"]),
            render_table(patient_stats, max_rows=10),
        ),
        "</div>",
        "<div class='panel'><h3>事件统计预览</h3>{}</div>".format(render_table(event_stats, max_rows=10)),
        "<div class='image-grid'>",
        inline_image(FOLD2_DIR / "windows_per_patient.png", 1100, "各病人的窗口数量分布"),
        inline_image(FOLD2_DIR / "seizures_per_patient.png", 1100, "各病人的 seizure 事件数量"),
        inline_image(FOLD2_DIR / "interval_length_distribution.png", 1100, "事件/区间长度分布"),
        inline_image(FOLD2_DIR / "sample_windows.png", 1200, "代表性 EEG 窗口样例"),
        "</div>",
        "<h3>当前产物</h3>",
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
        "<h2>三种实验协议与切分清洗</h2>",
        "<p>这一步把实验设计真正落成可复现的 split artifacts。它明确区分了 <code>window</code>、<code>seizure</code>、<code>patient</code> 三种泛化层级，并记录了和 negative sampling 相关的实际数据条件。</p>",
        fold3_cards,
        "<div class='two-col'>",
        "<div class='panel'><h3>Split QC 报告</h3>{}</div>".format(split_qc_html),
        "<div class='panel'><h3>window overlap 摘要</h3>{}</div>".format(render_table(window_overlap, max_rows=15)),
        "</div>",
        "<div class='two-col'>",
        "<div class='panel'><h3>window manifest 预览</h3>{}</div>".format(render_table(window_manifest_preview)),
        "<div class='panel'><h3>seizure manifest 预览</h3>{}</div>".format(render_table(seizure_manifest_preview)),
        "</div>",
        "<div class='panel'><h3>patient manifest 预览</h3>{}</div>".format(render_table(patient_manifest_preview)),
        note_box(
            "协议解释",
            "- `window`：train/test 允许共享 patient 与 event 上下文，因此它是 optimistic upper bound。\n"
            "- `seizure`：以 `(patient_id, global_interval)` 为单元留出未知发作事件。\n"
            "- `patient`：Leave-One-Subject-Out，是最严格的跨病人泛化。",
        ),
        "<h3>当前产物</h3>",
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
        "<h2>输入管线与病人级缓存</h2>",
        "<p>这一步的核心不是“又做了个图”，而是把原始 <code>npz</code> 读取转成稳定、可复用的 <code>float32</code> cache。这样后续多卡训练时就不需要每次重新反序列化整份病人数据，CPU 压力会更可控。</p>",
        fold4_cards,
        "<div class='two-col'>",
        "<div class='panel'><h3>cache 概览</h3>{}</div>".format(render_table(cache_df, max_rows=12)),
        "<div class='panel'><h3>这一步现在真实落盘了什么</h3><p>当前磁盘上稳定存在的是 24 份病人级 <code>.npy</code> cache 和 24 份对应 <code>.meta.json</code>。这一版静态报告按真实文件写，不把未单独落盘的中间统计硬说成独立产物。</p><p>缓存张量统一是 <code>[n_windows, 21, 128]</code>，dtype 为 <code>float32</code>。</p></div>",
        "</div>",
        "<h3>当前产物</h3>",
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
        "<h2>CNN 训练与 GPU 任务矩阵</h2>",
        "<p>这一步是你要求的 tmux + GPU 正式矩阵。训练入口不是 notebook，而是脚本调度到 GPU 上执行。当前 full run 已经全部完成，<code>0 failed folds</code>，说明从训练到保存 checkpoint 的全链路是通的。</p>",
        fold5_cards,
        note_box(
            "当前训练事实",
            "- 共有 10 个 full shard。\n"
            "- `window` 两个 train mode 各 1 个 shard。\n"
            "- `seizure` 两个 train mode 各拆成 3 个 shard。\n"
            "- `patient` 两个 train mode 各 1 个 shard。\n"
            "- 总共生成了 440 个 `best.pt` 和 440 份 `train_log.csv`。",
        ),
        "<div class='panel'><h3>full GPU run 状态表</h3>{}</div>".format(render_table(runtime_df, max_rows=20)),
        "<h3>当前产物</h3>",
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
        "<h2>统一评估与双测试分布</h2>",
        "<p>这一步把所有训练好的模型统一评估成可比较的指标文件。关键点是：<code>balanced</code> 和 <code>unbalanced</code> 的测试并没有重新训练模型，而是在同一个 checkpoint 上分别生成两套测试结果。</p>",
        fold6_cards,
        note_box(
            "评估逻辑",
            "- 阈值先在 validation 上选择，再应用到 test。\n"
            "- 每个 outer fold 都同时输出 `balanced_50_50` 和 `unbalanced_20_80` 两种 test 结果。\n"
            "- 因此 440 个训练模型最终对应了 880 份 `metrics.json` 和 880 份 `predictions.parquet`。",
        ),
        "<div class='panel'><h3>评估组合主览</h3>{}</div>".format(
            render_table(
                main_results[
                    ["protocol", "train_mode", "test_mode", "n_folds", "f1_mean", "roc_auc_mean", "precision_mean", "recall_mean", "specificity_mean"]
                ],
                max_rows=12,
                float_cols=["f1_mean", "roc_auc_mean", "precision_mean", "recall_mean", "specificity_mean"],
            )
        ),
        "<h3>当前产物</h3>",
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
        "<h2>结果汇总、科研风可视化与报告材料</h2>",
        "<p>这一步把前面所有实验真正整理成“能讲出去”的东西。这里包括主结果表、summary figures、research visualizations，以及你们报告里会直接引用的讨论文字。</p>",
        fold7_cards,
        "<div class='panel'><h3>主结果表</h3>{}</div>".format(
            render_table(
                main_results[
                    ["protocol", "train_mode", "test_mode", "n_folds", "f1_mean", "roc_auc_mean", "precision_mean", "recall_mean", "specificity_mean"]
                ],
                max_rows=12,
                float_cols=["f1_mean", "roc_auc_mean", "precision_mean", "recall_mean", "specificity_mean"],
            )
        ),
        "<div class='image-grid'>",
        inline_image(SUMMARY_DIR / "f1_by_protocol.png", 1150, "summary figure: 各协议的平均 F1"),
        inline_image(SUMMARY_DIR / "auc_by_protocol.png", 1150, "summary figure: 各协议的平均 ROC-AUC"),
        "</div>",
        "<div class='two-col'>",
        "<div class='panel'><h3>中文图注</h3>{}</div>".format(captions_html),
        "<div class='panel'><h3>可视化说明</h3>{}</div>".format(visual_notes_html),
        "</div>",
        "<div class='image-grid'>",
        inline_image(VIS_DIR / "input_heatmap_pair.png", 1200, "图 1. 单窗口 EEG 输入热力图对比"),
        inline_image(VIS_DIR / "channel_fusion_architecture.png", 1250, "图 2. Channel Fusion CNN 结构示意图"),
        inline_image(VIS_DIR / "first_layer_channel_fusion_weights.png", 1250, "图 3. 第一层卷积核的通道融合权重"),
        inline_image(VIS_DIR / "saliency_case_patient_fold0.png", 1250, "图 4. 发作样本的 Saliency 可解释性结果"),
        "</div>",
        "<div class='two-col'>",
        "<div class='panel'><h3>Applications Discussion</h3>{}</div>".format(discussion_html),
        "<div class='panel'><h3>Report-ready Notes</h3>{}</div>".format(report_notes_html),
        "</div>",
        "<h3>当前产物</h3>",
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
        "<h2>未来模型扩展位</h2>",
        "<p>这一步现在不训练任何新模型，但它把未来扩展的边界钉住了。也就是说，之后如果你们要补第二模型，不应该改 split schema 和 eval schema，而是直接复用前面已经稳定的工程接口。</p>",
        fold8_cards,
        "<div class='two-col'>",
        "<div class='panel'><h3>Future Model Slot</h3>{}</div>".format(future_readme_html),
        "<div class='panel'><h3>Model IO Contract</h3>{}</div>".format(future_contract_html),
        "</div>",
        "<h3>当前产物</h3>",
        artifact_list(
            [
                FOLD8_DIR / "future_model_readme.md",
                FOLD8_DIR / "model_io_contract.md",
            ]
        ),
        "</section>",

        "<div class='footer'>这个 HTML 已经替换掉之前那份省略过多的简版静态报告。现在发给同学时，只发这一个文件即可；里面的图片已经内嵌，不依赖 notebook 或旁边的文件夹。</div>",
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
