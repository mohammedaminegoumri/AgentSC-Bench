"""
Aggregate results, compute statistics, and produce publication-quality figures.

Usage:
    python -m experiments.evaluate --results-dir results/smoke
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)


def load_summaries(results_dir: Path) -> pd.DataFrame:
    frames = []
    for p in results_dir.glob("*_summary.csv"):
        frames.append(pd.read_csv(p))
    if not frames:
        raise FileNotFoundError(f"No *_summary.csv found in {results_dir}")
    return pd.concat(frames, ignore_index=True)


def load_timeseries(results_dir: Path) -> pd.DataFrame:
    frames = []
    for p in results_dir.glob("*_timeseries.csv"):
        frames.append(pd.read_csv(p))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["total_cost", "mean_service_level", "bullwhip_mean"]
    present = [m for m in metrics if m in df.columns]
    if not present:
        return df
    grouped = df.groupby(["architecture", "scenario", "model"], dropna=False)[present]
    stats = grouped.agg(["mean", "std", "count"]).round(3)
    return stats


def plot_inventory_over_time(ts: pd.DataFrame, out: Path, experiment_id: Optional[str] = None) -> None:
    if ts.empty:
        return
    if experiment_id:
        ts = ts[ts["experiment_id"] == experiment_id]
    fig, ax = plt.subplots(figsize=(8, 4))
    for ech in ts["echelon"].unique():
        sub = ts[ts["echelon"] == ech]
        # average across seeds/episodes
        g = sub.groupby("period")["inventory"].mean()
        ax.plot(g.index, g.values, label=ech)
    ax.set_xlabel("Period")
    ax.set_ylabel("Inventory")
    ax.set_title("Inventory over time (mean across seeds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "inventory_over_time.png", dpi=200)
    fig.savefig(out / "inventory_over_time.svg")
    plt.close(fig)


def plot_orders_over_time(ts: pd.DataFrame, out: Path) -> None:
    if ts.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    for ech in ts["echelon"].unique():
        sub = ts[ts["echelon"] == ech]
        g = sub.groupby("period")["order"].mean()
        ax.plot(g.index, g.values, label=ech)
    # demand
    dem = ts.groupby("period")["demand"].mean()
    ax.plot(dem.index, dem.values, "k--", label="customer demand", alpha=0.7)
    ax.set_xlabel("Period")
    ax.set_ylabel("Order quantity")
    ax.set_title("Orders over time")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "orders_over_time.png", dpi=200)
    fig.savefig(out / "orders_over_time.svg")
    plt.close(fig)


def plot_cost_comparison(summary: pd.DataFrame, out: Path) -> None:
    if "total_cost" not in summary.columns:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    order = summary.groupby("architecture")["total_cost"].mean().sort_values().index
    sns.barplot(data=summary, x="architecture", y="total_cost", order=order, ax=ax, errorbar="sd")
    ax.set_ylabel("Total cost")
    ax.set_title("Total cost by architecture")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out / "cost_comparison.png", dpi=200)
    fig.savefig(out / "cost_comparison.svg")
    plt.close(fig)


def plot_bullwhip_by_echelon(ts: pd.DataFrame, out: Path) -> None:
    if ts.empty:
        return
    # Compute order variance per echelon
    demand_var = ts.groupby("period")["demand"].mean().var()
    rows = []
    for ech in ts["echelon"].unique():
        ov = ts[ts["echelon"] == ech].groupby("period")["order"].mean().var()
        ratio = ov / demand_var if demand_var > 1e-9 else 0.0
        rows.append({"echelon": ech, "bullwhip_ratio": ratio})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=df, x="echelon", y="bullwhip_ratio", ax=ax)
    ax.set_title("Classical bullwhip ratio by echelon")
    ax.axhline(1.0, color="gray", ls="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out / "bullwhip_by_echelon.png", dpi=200)
    fig.savefig(out / "bullwhip_by_echelon.svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--out", type=str, default=None, help="Figure output dir (default: results/figures)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out = Path(args.out) if args.out else results_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)

    summary = load_summaries(results_dir)
    ts = load_timeseries(results_dir)

    print("=== Summary statistics ===")
    print(summary_table(summary))

    plot_inventory_over_time(ts, out)
    plot_orders_over_time(ts, out)
    plot_cost_comparison(summary, out)
    plot_bullwhip_by_echelon(ts, out)

    # Save aggregated table
    summary.to_csv(out / "aggregated_summary.csv", index=False)
    print(f"\nFigures written to {out}")


if __name__ == "__main__":
    main()
