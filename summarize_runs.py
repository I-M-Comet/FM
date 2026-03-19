#!/usr/bin/env python3
"""
Summarize EEG run metrics (JSONL) and LP results (CSV).

Features
- Metric-side summary from training JSONL logs.
- LP-side summary from eval CSV.
- Best run by metric proxy (late-window heuristic).
- Best run by LP (chance-normalized mean val acc).
- Task-wise winners.
- Recipe-level aggregation with simple seed stripping.

Example
-------
python summarize_runs.py --metrics A4_1_runs_metrics.jsonl --lp eval_A4_1_lp_results.csv --output output.txt
python summarize_eeg_runs.py \
  --metrics /mnt/data/A3_6_runs_metrics.jsonl \
  --lp /mnt/data/eval_A3_6_lp_results.csv \
  --output /mnt/data/A3_6_summary.txt
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Helpers
# -----------------------------


def read_jsonl(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at line {line_no} in {path}: {e}")
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def derive_run_key_from_ckpt_path(path: str) -> str:
    if not isinstance(path, str) or not path:
        return ""
    # e.g. /mnt/e/checkpoints/A3_6_qkbranch/final/teacher -> A3_6_qkbranch
    norm = path.replace("\\", "/")
    m = re.search(r"/checkpoints/([^/]+)/", norm)
    if m:
        return m.group(1)
    parts = [p for p in norm.split("/") if p]
    if len(parts) >= 2:
        return parts[-2]
    return os.path.basename(norm)


def normalize_run_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    name = name.strip()
    name = re.sub(r"^eval_", "", name)
    name = re.sub(r"^teacher_", "", name)
    return name


def strip_seed_tokens(name: str) -> str:
    """Best-effort stripping of seed tokens for recipe-level aggregation."""
    if not isinstance(name, str):
        return ""
    s = normalize_run_name(name)
    patterns = [
        r"([_-])sd\d+\b",
        r"([_-])seed\d+\b",
        r"\bsd\d+\b",
        r"\bseed\d+\b",
        r"([_-])s\d+\b",
    ]
    for p in patterns:
        s = re.sub(p, "", s, flags=re.IGNORECASE)
    s = re.sub(r"__+", "_", s)
    s = re.sub(r"--+", "-", s)
    s = re.sub(r"[_-]+$", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def chance_from_row(row: pd.Series) -> float:
    if "n_classes" in row and pd.notna(row["n_classes"]):
        try:
            n = int(row["n_classes"])
            if n > 0:
                return 1.0 / n
        except Exception:
            pass
    task = str(row.get("task", "")).lower()
    mapping = {"tuab": 0.5, "mi": 0.25, "isruc": 0.2}
    return mapping.get(task, np.nan)


def cn_score(acc: float, chance: float) -> float:
    if pd.isna(acc) or pd.isna(chance):
        return np.nan
    denom = 1.0 - chance
    if denom <= 0:
        return np.nan
    return (acc - chance) / denom


def safe_mean(series: pd.Series) -> float:
    series = pd.to_numeric(series, errors="coerce")
    if len(series.dropna()) == 0:
        return np.nan
    return float(series.mean())


def fmt(x: float, digits: int = 4) -> str:
    if x is None or pd.isna(x):
        return "nan"
    return f"{x:.{digits}f}"


def make_rank(series: pd.Series, ascending: bool) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return s.rank(method="average", ascending=ascending, na_option="bottom")


# -----------------------------
# Metrics summary
# -----------------------------

@dataclass
class MetricsConfig:
    late_fraction: float = 0.2
    min_late_points: int = 10


def summarize_metrics(metrics_df: pd.DataFrame, cfg: MetricsConfig) -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame()
    if "run_name" not in metrics_df.columns:
        raise ValueError("Metrics JSONL must contain 'run_name'.")
    if "_step" not in metrics_df.columns:
        metrics_df = metrics_df.copy()
        metrics_df["_step"] = np.arange(len(metrics_df))

    rows = []
    numeric_cols = [
        "proxy/cos_mean",
        "proxy/cos_mean_last",
        "loss_tgt_mean",
        "loss_scaled",
        "proxy/pred_feat_std_last",
        "proxy/tgt_feat_std_last",
        "proxy/pred_norm_mean",
        "proxy/tgt_norm_mean",
        "proxy/pred_norm_std",
        "proxy/tgt_norm_std",
        "grad_norm",
        "ctx_tokens_sum",
        "tgt_tokens_sum",
    ]
    existing_numeric = [c for c in numeric_cols if c in metrics_df.columns]

    for run_name, g in metrics_df.groupby("run_name", sort=False):
        g = g.sort_values("_step").reset_index(drop=True)
        n = len(g)
        late_n = max(cfg.min_late_points, int(math.ceil(n * cfg.late_fraction)))
        late_n = min(late_n, n)
        late = g.tail(late_n)

        row: Dict[str, float | str | int] = {
            "run_name": str(run_name),
            "recipe": strip_seed_tokens(str(run_name)),
            "n_points": int(n),
            "max_step": int(pd.to_numeric(g["_step"], errors="coerce").max()),
        }
        for c in existing_numeric:
            row[f"{c}__last"] = pd.to_numeric(g[c], errors="coerce").dropna().iloc[-1] if len(pd.to_numeric(g[c], errors="coerce").dropna()) else np.nan
            row[f"{c}__late_mean"] = safe_mean(pd.to_numeric(late[c], errors="coerce"))
            row[f"{c}__overall_mean"] = safe_mean(pd.to_numeric(g[c], errors="coerce"))
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # Heuristic metric score.
    cos_col = "proxy/cos_mean__late_mean"
    loss_col = "loss_tgt_mean__late_mean"
    pred_std_col = "proxy/pred_feat_std_last__late_mean"
    tgt_std_col = "proxy/tgt_feat_std_last__late_mean"

    for c in [cos_col, loss_col, pred_std_col, tgt_std_col]:
        if c not in out.columns:
            out[c] = np.nan

    pred_std_err = (pd.to_numeric(out[pred_std_col], errors="coerce") - 1.0).abs()
    tgt_std_err = (pd.to_numeric(out[tgt_std_col], errors="coerce") - 1.0).abs()

    # Rank-based composite is robust to scale.
    score = pd.Series(0.0, index=out.index)
    score += make_rank(out[cos_col], ascending=False)          # higher cos is better
    score += make_rank(out[loss_col], ascending=True)          # lower loss is better
    score += make_rank(pred_std_err, ascending=True)          # closer to 1 is better
    score += make_rank(tgt_std_err, ascending=True)           # closer to 1 is better

    out["metric_proxy_score"] = score
    out["pred_std_err"] = pred_std_err
    out["tgt_std_err"] = tgt_std_err

    # Lower rank sum is better. Convert to descending convenience score too.
    out["metric_proxy_rank"] = out["metric_proxy_score"].rank(method="average", ascending=True)
    out = out.sort_values(["metric_proxy_score", cos_col], ascending=[True, False]).reset_index(drop=True)
    return out


# -----------------------------
# LP summary
# -----------------------------


def summarize_lp(lp_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if lp_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = lp_df.copy()
    if "ckpt_path" in df.columns:
        df["run_key"] = df["ckpt_path"].map(derive_run_key_from_ckpt_path)
    else:
        df["run_key"] = df.get("run_name", "")
    df["run_key"] = df["run_key"].map(normalize_run_name)
    if "run_name" in df.columns:
        # Prefer ckpt-derived key if available; fallback to run_name.
        df["run_key"] = np.where(df["run_key"].eq(""), df["run_name"].map(normalize_run_name), df["run_key"])

    df["recipe"] = df["run_key"].map(strip_seed_tokens)
    df["chance"] = df.apply(chance_from_row, axis=1)
    df["cn_val_acc"] = [cn_score(a, c) for a, c in zip(df["val_acc"], df["chance"])]
    df["cn_test_acc"] = [cn_score(a, c) for a, c in zip(df["test_acc"], df["chance"])]

    # Run-level summary.
    agg = {
        "val_acc": "mean",
        "val_f1w": "mean",
        "test_acc": "mean",
        "test_f1w": "mean",
        "cn_val_acc": "mean",
        "cn_test_acc": "mean",
        "task": lambda s: ",".join(sorted(map(str, s.unique()))),
    }
    run_summary = df.groupby(["run_key", "recipe"], as_index=False).agg(agg)
    run_summary = run_summary.rename(columns={"task": "tasks"})
    run_summary = run_summary.sort_values(["cn_val_acc", "test_acc"], ascending=[False, False]).reset_index(drop=True)

    # Recipe-level summary.
    recipe_summary = df.groupby("recipe", as_index=False).agg(
        n_rows=("task", "size"),
        n_tasks=("task", lambda s: len(set(map(str, s)))),
        val_acc=("val_acc", "mean"),
        val_f1w=("val_f1w", "mean"),
        test_acc=("test_acc", "mean"),
        test_f1w=("test_f1w", "mean"),
        cn_val_acc=("cn_val_acc", "mean"),
        cn_test_acc=("cn_test_acc", "mean"),
        n_runs=("run_key", lambda s: len(set(map(str, s)))),
    )
    recipe_summary = recipe_summary.sort_values(["cn_val_acc", "test_acc"], ascending=[False, False]).reset_index(drop=True)

    return run_summary, recipe_summary


# -----------------------------
# Text report
# -----------------------------


def df_to_pretty_table(df: pd.DataFrame, cols: List[str], top_k: int) -> str:
    if df.empty:
        return "<empty>"
    show = df.loc[:, [c for c in cols if c in df.columns]].head(top_k).copy()
    for c in show.columns:
        if pd.api.types.is_numeric_dtype(show[c]):
            show[c] = show[c].map(lambda x: fmt(x, 4))
    return show.to_string(index=False)


def task_winners(lp_df: pd.DataFrame) -> pd.DataFrame:
    if lp_df.empty:
        return pd.DataFrame()
    df = lp_df.copy()
    if "ckpt_path" in df.columns:
        df["run_key"] = df["ckpt_path"].map(derive_run_key_from_ckpt_path)
    else:
        df["run_key"] = df.get("run_name", "")
    df["run_key"] = df["run_key"].map(normalize_run_name)
    df["chance"] = df.apply(chance_from_row, axis=1)
    df["cn_val_acc"] = [cn_score(a, c) for a, c in zip(df["val_acc"], df["chance"])]
    df["cn_test_acc"] = [cn_score(a, c) for a, c in zip(df["test_acc"], df["chance"])]

    rows = []
    for task, g in df.groupby("task"):
        gv = g.sort_values(["cn_val_acc", "val_acc"], ascending=[False, False]).iloc[0]
        gt = g.sort_values(["cn_test_acc", "test_acc"], ascending=[False, False]).iloc[0]
        rows.append({
            "task": task,
            "best_val_run": gv["run_key"],
            "best_val_acc": gv["val_acc"],
            "best_val_cn": gv["cn_val_acc"],
            "best_test_run": gt["run_key"],
            "best_test_acc": gt["test_acc"],
            "best_test_cn": gt["cn_test_acc"],
        })
    out = pd.DataFrame(rows)
    return out.sort_values("task").reset_index(drop=True)


# -----------------------------
# Main
# -----------------------------


def build_report(metrics_path: Optional[str], lp_path: Optional[str], late_fraction: float, min_late_points: int, top_k: int) -> str:
    buf = io.StringIO()
    print("EEG run summary", file=buf)
    print("=" * 80, file=buf)
    if metrics_path:
        print(f"metrics: {metrics_path}", file=buf)
    if lp_path:
        print(f"lp:      {lp_path}", file=buf)
    print(file=buf)

    metrics_summary = pd.DataFrame()
    if metrics_path:
        metrics_df = read_jsonl(metrics_path)
        metrics_summary = summarize_metrics(metrics_df, MetricsConfig(late_fraction=late_fraction, min_late_points=min_late_points))
        print("[Metric-side summary]", file=buf)
        if metrics_summary.empty:
            print("No metric rows found.\n", file=buf)
        else:
            best = metrics_summary.iloc[0]
            print(f"Best metric proxy run: {best['run_name']}", file=buf)
            print(f"  metric_proxy_score(rank-sum, lower better): {fmt(best['metric_proxy_score'])}", file=buf)
            print(f"  late cos_mean:    {fmt(best.get('proxy/cos_mean__late_mean', np.nan))}", file=buf)
            print(f"  late loss_tgt:    {fmt(best.get('loss_tgt_mean__late_mean', np.nan))}", file=buf)
            print(f"  pred_feat_std:    {fmt(best.get('proxy/pred_feat_std_last__late_mean', np.nan))}", file=buf)
            print(f"  tgt_feat_std:     {fmt(best.get('proxy/tgt_feat_std_last__late_mean', np.nan))}", file=buf)
            print(file=buf)
            print(df_to_pretty_table(
                metrics_summary,
                cols=[
                    "run_name", "recipe", "metric_proxy_score",
                    "proxy/cos_mean__late_mean", "loss_tgt_mean__late_mean",
                    "proxy/pred_feat_std_last__late_mean", "proxy/tgt_feat_std_last__late_mean",
                    "max_step", "n_points"
                ],
                top_k=top_k,
            ), file=buf)
            print(file=buf)

    lp_run_summary = pd.DataFrame()
    lp_recipe_summary = pd.DataFrame()
    winners = pd.DataFrame()
    if lp_path:
        lp_df = pd.read_csv(lp_path)
        lp_run_summary, lp_recipe_summary = summarize_lp(lp_df)
        winners = task_winners(lp_df)
        print("[LP-side summary]", file=buf)
        if lp_run_summary.empty:
            print("No LP rows found.\n", file=buf)
        else:
            best = lp_run_summary.iloc[0]
            print(f"Best LP run (by chance-normalized mean val acc): {best['run_key']}", file=buf)
            print(f"  cn mean val acc:  {fmt(best['cn_val_acc'])}", file=buf)
            print(f"  cn mean test acc: {fmt(best['cn_test_acc'])}", file=buf)
            print(f"  mean val acc:     {fmt(best['val_acc'])}", file=buf)
            print(f"  mean test acc:    {fmt(best['test_acc'])}", file=buf)
            print(file=buf)
            print(df_to_pretty_table(
                lp_run_summary,
                cols=["run_key", "recipe", "cn_val_acc", "cn_test_acc", "val_acc", "test_acc", "val_f1w", "test_f1w", "tasks"],
                top_k=top_k,
            ), file=buf)
            print(file=buf)

            print("[Task-wise LP winners]", file=buf)
            print(df_to_pretty_table(
                winners,
                cols=["task", "best_val_run", "best_val_acc", "best_val_cn", "best_test_run", "best_test_acc", "best_test_cn"],
                top_k=top_k,
            ), file=buf)
            print(file=buf)

            print("[Recipe-level LP summary]", file=buf)
            print(df_to_pretty_table(
                lp_recipe_summary,
                cols=["recipe", "n_runs", "cn_val_acc", "cn_test_acc", "val_acc", "test_acc", "val_f1w", "test_f1w"],
                top_k=top_k,
            ), file=buf)
            print(file=buf)

    if not metrics_summary.empty and not lp_run_summary.empty:
        # Attempt to compare overlapping runs.
        merged = metrics_summary.merge(lp_run_summary, left_on="run_name", right_on="run_key", how="inner")
        if merged.empty:
            # fallback by normalized name
            m2 = metrics_summary.copy()
            m2["join_key"] = m2["run_name"].map(normalize_run_name)
            l2 = lp_run_summary.copy()
            l2["join_key"] = l2["run_key"].map(normalize_run_name)
            merged = m2.merge(l2, on="join_key", how="inner", suffixes=("_metric", "_lp"))
        if not merged.empty:
            print("[Overlap: metric vs LP winners among matched runs]", file=buf)
            # Pick best LP among matched runs
            if "cn_val_acc" in merged.columns:
                best_lp = merged.sort_values(["cn_val_acc", "test_acc"], ascending=[False, False]).iloc[0]
                print(f"Best matched LP run: {best_lp.get('run_key', best_lp.get('run_name', ''))}", file=buf)
                print(f"  cn mean val acc:  {fmt(best_lp['cn_val_acc'])}", file=buf)
                print(f"  mean test acc:    {fmt(best_lp['test_acc'])}", file=buf)
            score_col = "metric_proxy_score"
            if score_col in merged.columns:
                best_metric = merged.sort_values([score_col], ascending=[True]).iloc[0]
                print(f"Best matched metric run: {best_metric.get('run_name', best_metric.get('run_key', ''))}", file=buf)
                print(f"  metric proxy score: {fmt(best_metric[score_col])}", file=buf)
            print(file=buf)

    return buf.getvalue()


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize EEG run metrics and LP results.")
    p.add_argument("--metrics", type=str, default=None, help="Path to metrics JSONL.")
    p.add_argument("--lp", type=str, default=None, help="Path to LP CSV.")
    p.add_argument("--output", type=str, default=None, help="Optional output txt path.")
    p.add_argument("--late-fraction", type=float, default=0.2, help="Fraction of tail points to average for late-window metrics.")
    p.add_argument("--min-late-points", type=int, default=10, help="Minimum number of tail points in late-window metrics.")
    p.add_argument("--top-k", type=int, default=50, help="How many rows to print in each section.")
    args = p.parse_args(argv)
    if not args.metrics and not args.lp:
        p.error("At least one of --metrics or --lp must be provided.")
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    report = build_report(
        metrics_path=args.metrics,
        lp_path=args.lp,
        late_fraction=args.late_fraction,
        min_late_points=args.min_late_points,
        top_k=args.top_k,
    )
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report)
    sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
