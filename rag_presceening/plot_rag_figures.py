#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figures from RAG corpus (processed_papers.json) and batch QA (batch_qa / final.json).
Outputs go under data/rag_figures/.

Toggle PLOT_* at the top to skip figures; if all corpus plots are off, processed_papers.json is not read;
if all batch plots are off, batch JSON is not read.

From project root:
  python -m rag_presceening.plot_rag_figures
  python -m rag_presceening.plot_rag_figures --batch rag_presceening/data/rag_runs/final.json --no-show
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# PLOT_* flags (False = skip). Outputs under --out-dir.
# PLOT_CORPUS_OVERVIEW: rag_corpus_pages.png, rag_corpus_text_length.png, rag_corpus_keywords.png
# PLOT_CORPUS_EXTRACTION_PIE: rag_corpus_extraction_method.png
# PLOT_RETRIEVAL_DIAGNOSTICS: rag_retrieval_dense_similarity_box.png, rag_retrieval_ranking_score_box.png
# PLOT_TAG_COVERAGE_HEATMAP: rag_tag_coverage_heatmap.png
# PLOT_QUERY_LABELS_TXT: rag_query_labels.txt (Q id + first line of each question)
# =============================================================================
PLOT_CORPUS_OVERVIEW = False
PLOT_CORPUS_EXTRACTION_PIE = False
PLOT_RETRIEVAL_DIAGNOSTICS = False
PLOT_TAG_COVERAGE_HEATMAP = True
PLOT_QUERY_LABELS_TXT = True

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica", "sans-serif"],
        "font.size": 20,
        "axes.titlesize": 20,
        "axes.labelsize": 20,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 20,
        "axes.linewidth": 0.8,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

COLORS = {
    "primary": "#2E86AB",
    "secondary": "#A23B72",
    "accent": "#2ca02c",
    "neutral": "#6c757d",
    "stack": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"],
}

SAVE_DPI = 300

FIGSIZE_SINGLE = (6.2, 4.6)
FIGSIZE_KEYWORDS = (6.2, 5.2)
FONT_SIZE_CORPUS_KEYWORDS = 20

FONT_SIZE_TAG_HEATMAP_ROW = 20
FONT_SIZE_TAG_HEATMAP_COL = 20
FONT_SIZE_TAG_HEATMAP_XLABEL = 20
FONT_SIZE_TAG_HEATMAP_TITLE = 20
FONT_SIZE_TAG_HEATMAP_CBAR = 20


def _finalize_figure(fig: plt.Figure, out_path: Path, show: bool) -> None:
    fig.tight_layout()
    fig.savefig(out_path, dpi=SAVE_DPI)
    print(f"Saved: {out_path}")
    if show:
        plt.show()
    plt.close(fig)


def _base_dir() -> Path:
    return Path(__file__).resolve().parent


def load_processed_papers(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing corpus JSON: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("processed_papers.json must be a JSON array")
    return data


def load_batch_records(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing batch JSON: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("batch file must be a JSON array")
    return data


def question_label(rec: Dict[str, Any], max_len: int = 72) -> str:
    """Q index + first-line summary for captions."""
    idx = rec.get("index", 0)
    q = (rec.get("question") or "").strip()
    first = q.split("\n", 1)[0].strip()
    if len(first) > max_len:
        first = first[: max_len - 1] + "…"
    return f"Q{idx}: {first}"


def question_label_short(rec: Dict[str, Any]) -> str:
    return f"Q{rec.get('index', 0)}"


def parse_tags(tags_field: Any) -> List[str]:
    if not tags_field:
        return []
    s = str(tags_field).strip()
    parts = re.split(r"[;]", s)
    return [p.strip() for p in parts if p.strip()]


def plot_corpus_overview(
    papers: List[Dict[str, Any]],
    out_dir: Path,
    show: bool,
) -> None:
    """Corpus overview and/or extraction pie per top-of-file flags."""
    if not (PLOT_CORPUS_OVERVIEW or PLOT_CORPUS_EXTRACTION_PIE):
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    pages: List[int] = []
    lengths: List[int] = []
    methods: List[str] = []
    kw_counter: Counter[str] = Counter()

    for doc in papers:
        meta = doc.get("metadata") or {}
        tp = meta.get("total_pages")
        if tp is not None:
            try:
                pages.append(int(tp))
            except (TypeError, ValueError):
                pass
        tl = doc.get("text_length")
        if tl is not None:
            try:
                lengths.append(int(tl))
            except (TypeError, ValueError):
                pass
        em = meta.get("extraction_method") or "unknown"
        methods.append(str(em))
        for kw in doc.get("keywords") or []:
            if isinstance(kw, str) and kw.strip():
                kw_counter[kw.strip().lower()] += 1

    if PLOT_CORPUS_OVERVIEW:
        _plot_corpus_overview_figure(
            papers, pages, lengths, kw_counter, out_dir, show
        )

    if PLOT_CORPUS_EXTRACTION_PIE:
        _plot_corpus_extraction_pie(methods, out_dir, show)


def _plot_corpus_overview_figure(
    papers: List[Dict[str, Any]],
    pages: List[int],
    lengths: List[int],
    kw_counter: Counter,
    out_dir: Path,
    show: bool,
) -> None:
    """Three PNGs: pages, text length, keywords."""

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    if pages:
        ax.hist(
            pages,
            bins=min(20, max(5, len(set(pages)))),
            color=COLORS["primary"],
            edgecolor="white",
            linewidth=0.5,
        )
        ax.set_xlabel("Pages per paper")
        ax.set_ylabel("Count")
        ax.set_title(f"Corpus pages (N={len(papers)} papers)")
    else:
        ax.text(0.5, 0.5, "No total_pages in metadata", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Corpus pages")
    _finalize_figure(fig, out_dir / "rag_corpus_pages.png", show)

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    if lengths:
        kb = np.array(lengths) / 1000.0
        ax.hist(kb, bins=20, color=COLORS["secondary"], edgecolor="white", linewidth=0.5)
        ax.set_xlabel("Text length (×10³ characters)")
        ax.set_ylabel("Count")
        ax.set_title("Extracted text length per paper")
    else:
        ax.text(0.5, 0.5, "No text_length", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Extracted text length")
    _finalize_figure(fig, out_dir / "rag_corpus_text_length.png", show)

    top_n = 15
    top_kw = kw_counter.most_common(top_n)
    fig, ax = plt.subplots(figsize=FIGSIZE_KEYWORDS)
    if top_kw:
        kw_labels = [k for k, _ in reversed(top_kw)]
        vals = [v for _, v in reversed(top_kw)]
        y = np.arange(len(kw_labels))
        ax.barh(y, vals, color=COLORS["accent"], height=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels(kw_labels, fontsize=FONT_SIZE_CORPUS_KEYWORDS)
        ax.set_xlabel("Paper count (keyword hit)")
        ax.set_title(f"Top {top_n} corpus keywords")
    else:
        ax.text(0.5, 0.5, "No keywords", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Corpus keywords")
    _finalize_figure(fig, out_dir / "rag_corpus_keywords.png", show)


def _plot_corpus_extraction_pie(methods: List[str], out_dir: Path, show: bool) -> None:
    """rag_corpus_extraction_method.png: PDF backend mix."""
    mc = Counter(methods)
    if len(mc) > 1 or (len(mc) == 1 and list(mc.keys())[0] != "unknown"):
        fig2, ax2 = plt.subplots(figsize=(4.2, 4.2))
        names = list(mc.keys())
        vals = [mc[n] for n in names]
        ax2.pie(vals, labels=names, autopct="%1.0f%%", colors=COLORS["stack"][: len(names)])
        ax2.set_title("PDF text extraction method")
        _finalize_figure(fig2, out_dir / "rag_corpus_extraction_method.png", show)


def collect_retrieval_stats(records: List[Dict[str, Any]]) -> Tuple:
    """Aggregate scores, file counts, section counts, tag frequencies for plots."""
    labels = [question_label_short(r) for r in records]
    long_labels = [question_label(r) for r in records]

    scores_by_q: List[List[float]] = []
    rank_by_q: List[List[float]] = []
    n_unique_files: List[int] = []
    section_keys_order = ["introduction", "methods", "results", "conclusion", "abstract", "other", "unknown"]

    section_counts: Dict[str, List[int]] = {k: [] for k in section_keys_order}
    all_tags: Counter[str] = Counter()

    for rec in records:
        docs = rec.get("retrieved_documents") or []
        sc, rk = [], []
        files = set()
        sec_local: Counter[str] = Counter()
        for d in docs:
            meta = d.get("metadata") or {}
            s = d.get("score")
            if s is not None:
                try:
                    sc.append(float(s))
                except (TypeError, ValueError):
                    pass
            rs = d.get("ranking_score")
            if rs is not None:
                try:
                    rk.append(float(rs))
                except (TypeError, ValueError):
                    pass
            fn = meta.get("filename") or meta.get("original_title")
            if fn:
                files.add(str(fn))
            st = (meta.get("section_type") or "unknown").lower().strip()
            if st not in section_keys_order:
                st = "other"
            sec_local[st] += 1
            for t in parse_tags(meta.get("tags")):
                all_tags[t] += 1

        scores_by_q.append(sc)
        rank_by_q.append(rk if rk else sc)
        n_unique_files.append(len(files))

        for k in section_keys_order:
            section_counts[k].append(sec_local.get(k, 0))

    return (
        labels,
        long_labels,
        scores_by_q,
        rank_by_q,
        n_unique_files,
        section_keys_order,
        section_counts,
        all_tags,
    )


def plot_retrieval_analysis(
    records: List[Dict[str, Any]],
    out_dir: Path,
    show: bool,
    top_tags: int = 12,
) -> None:
    """Retrieval diagnostics, tag heatmap, and/or query label txt per flags."""
    if not records:
        print("No batch records; skip retrieval figures.")
        return

    if not (PLOT_RETRIEVAL_DIAGNOSTICS or PLOT_TAG_COVERAGE_HEATMAP or PLOT_QUERY_LABELS_TXT):
        return

    need_stats = PLOT_RETRIEVAL_DIAGNOSTICS or PLOT_TAG_COVERAGE_HEATMAP
    if need_stats:
        (
            labels,
            long_labels,
            scores_by_q,
            rank_by_q,
            n_unique_files,
            section_keys_order,
            section_counts,
            all_tags,
        ) = collect_retrieval_stats(records)
    else:
        labels = [question_label_short(r) for r in records]
        long_labels = [question_label(r) for r in records]

    out_dir.mkdir(parents=True, exist_ok=True)

    if PLOT_RETRIEVAL_DIAGNOSTICS:
        _plot_retrieval_diagnostics_figure(labels, scores_by_q, rank_by_q, out_dir, show)

    if PLOT_TAG_COVERAGE_HEATMAP:
        _plot_tag_coverage_heatmap(records, labels, all_tags, out_dir, show, top_tags)

    if PLOT_QUERY_LABELS_TXT:
        cap = out_dir / "rag_query_labels.txt"
        with open(cap, "w", encoding="utf-8") as f:
            for long in long_labels:
                f.write(f"{long}\n")
        print(f"Saved: {cap}")


def _retrieval_boxplot_on_ax(
    ax: plt.Axes,
    x: np.ndarray,
    q_labels: List[str],
    data_list: List[List[float]],
    ylabel: str,
    title: str,
    color: str,
) -> None:
    """Per-query boxplot: whiskers span min/max (showfliers=False)."""
    if not any(data_list):
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
    else:
        bp_kw = dict(positions=x, widths=0.5, patch_artist=True, showfliers=False)
        try:
            bp = ax.boxplot(data_list, tick_labels=q_labels, **bp_kw)
        except TypeError:
            bp = ax.boxplot(data_list, labels=q_labels, **bp_kw)
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)


def _plot_retrieval_diagnostics_figure(
    labels: List[str],
    scores_by_q: List[List[float]],
    rank_by_q: List[List[float]],
    out_dir: Path,
    show: bool,
) -> None:
    """Two PNGs: dense similarity and ranking score boxplots."""
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    _retrieval_boxplot_on_ax(
        ax,
        x,
        labels,
        scores_by_q,
        "Dense similarity (cosine)",
        "Retrieved-chunk similarity by query",
        COLORS["primary"],
    )
    _finalize_figure(fig, out_dir / "rag_retrieval_dense_similarity_box.png", show)

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    _retrieval_boxplot_on_ax(
        ax,
        x,
        labels,
        rank_by_q,
        "Ranking score (after rerank)",
        "Reranked score by query",
        COLORS["secondary"],
    )
    _finalize_figure(fig, out_dir / "rag_retrieval_ranking_score_box.png", show)


def _plot_tag_coverage_heatmap(
    records: List[Dict[str, Any]],
    labels: List[str],
    all_tags: Counter,
    out_dir: Path,
    show: bool,
    top_tags: int,
) -> None:
    """rag_tag_coverage_heatmap.png: tag hit fraction per query."""
    tag_names = [t for t, _ in all_tags.most_common(top_tags)]
    if not tag_names:
        return

    mat = np.zeros((len(tag_names), len(labels)))
    for j, rec in enumerate(records):
        docs = rec.get("retrieved_documents") or []
        n = max(len(docs), 1)
        for i, tag in enumerate(tag_names):
            c = 0
            for d in docs:
                meta = d.get("metadata") or {}
                if tag in parse_tags(meta.get("tags")):
                    c += 1
            mat[i, j] = c / n

    fig, ax_hm = plt.subplots(figsize=(max(5, 1.2 * len(labels) + 3), 0.35 * len(tag_names) + 2))
    im = ax_hm.imshow(mat, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax_hm.set_xticks(np.arange(len(labels)))
    ax_hm.set_xticklabels(labels, fontsize=FONT_SIZE_TAG_HEATMAP_COL)
    ax_hm.set_yticks(np.arange(len(tag_names)))
    ax_hm.set_yticklabels(tag_names, fontsize=FONT_SIZE_TAG_HEATMAP_ROW)
    ax_hm.set_xlabel("Batch question", fontsize=FONT_SIZE_TAG_HEATMAP_XLABEL)
    ax_hm.set_title("Chunk-level tag hit rate (per query)", fontsize=FONT_SIZE_TAG_HEATMAP_TITLE)
    cbar = fig.colorbar(im, ax=ax_hm, fraction=0.035, pad=0.02)
    cbar.ax.tick_params(labelsize=FONT_SIZE_TAG_HEATMAP_CBAR)
    cbar.set_label("Fraction of retrieved chunks", fontsize=FONT_SIZE_TAG_HEATMAP_CBAR)
    _finalize_figure(fig, out_dir / "rag_tag_coverage_heatmap.png", show)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot RAG corpus + retrieval figures for papers")
    parser.add_argument(
        "--papers",
        type=str,
        default=None,
        help="Path to processed_papers.json",
    )
    parser.add_argument(
        "--batch",
        type=str,
        default=None,
        help="Path to batch_qa_*.json or final.json",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for PNG files",
    )
    parser.add_argument("--no-show", action="store_true", help="Do not call plt.show()")
    args = parser.parse_args()

    base = _base_dir()
    papers_path = Path(args.papers) if args.papers else base / "data" / "processed_papers.json"
    batch_path = Path(args.batch) if args.batch else base / "data" / "rag_runs" / "final.json"
    out_dir = Path(args.out_dir) if args.out_dir else base / "data" / "rag_figures"
    show = not args.no_show

    need_corpus = PLOT_CORPUS_OVERVIEW or PLOT_CORPUS_EXTRACTION_PIE
    need_batch = PLOT_RETRIEVAL_DIAGNOSTICS or PLOT_TAG_COVERAGE_HEATMAP or PLOT_QUERY_LABELS_TXT

    if need_corpus:
        papers = load_processed_papers(papers_path)
        plot_corpus_overview(papers, out_dir, show=show)
    else:
        print("Skip corpus figures (all PLOT_CORPUS_* are False).")

    if need_batch:
        if batch_path.is_file():
            records = load_batch_records(batch_path)
            plot_retrieval_analysis(records, out_dir, show=show)
        else:
            print(f"Skip batch figures (file not found): {batch_path}")
    else:
        print("Skip batch figures (all PLOT_RETRIEVAL_* / PLOT_QUERY_LABELS_TXT are False).")

    print(f"\nAll figures under: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
