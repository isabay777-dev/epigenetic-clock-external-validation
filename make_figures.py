#!/usr/bin/env python3
"""Create publication-quality figures for the methylation-clock analysis."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def savefig(path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def figure_external_scatter(results_dir: Path, figures_dir: Path, dpi: int) -> None:
    df = pd.read_csv(results_dir / "external_predictions.csv")
    df = df.loc[df["model"] == "elastic_net_cv"].copy()
    directions = list(df.groupby(["train_dataset", "test_dataset"], sort=False).groups)
    fig, axes = plt.subplots(1, len(directions), figsize=(10, 4.2), sharex=False, sharey=False)
    if len(directions) == 1:
        axes = [axes]
    for ax, (train_label, test_label) in zip(axes, directions):
        frame = df.loc[(df["train_dataset"] == train_label) & (df["test_dataset"] == test_label)]
        ax.scatter(frame["chronological_age"], frame["predicted_age"], s=18, alpha=0.7, color="#2f6f73", edgecolor="none")
        lo = min(frame["chronological_age"].min(), frame["predicted_age"].min())
        hi = max(frame["chronological_age"].max(), frame["predicted_age"].max())
        ax.plot([lo, hi], [lo, hi], color="#333333", linewidth=1.1, linestyle="--")
        ax.set_title(f"Train {train_label}\nTest {test_label}", fontsize=10)
        ax.set_xlabel("Chronological age (years)")
        ax.set_ylabel("Predicted age (years)")
        ax.grid(alpha=0.18)
    fig.suptitle("External validation: ElasticNetCV predicted vs chronological age", fontsize=12)
    savefig(figures_dir / "external_predicted_vs_chronological.png", dpi)


def figure_mae_bar(results_dir: Path, figures_dir: Path, dpi: int) -> None:
    external = load_json(results_dir / "external_validation.json")["external_validation"]
    df = pd.DataFrame(external)
    models = ["plain_elastic_net_baseline", "elastic_net_cv", "hist_gradient_boosting", "mlp", "ensemble_average"]
    labels = ["Plain EN", "ElasticNetCV", "HGBR", "MLP", "Ensemble"]
    directions = list(df.groupby(["train_dataset", "test_dataset"], sort=False).groups)
    x = np.arange(len(directions))
    width = 0.15
    plt.figure(figsize=(10, 4.4))
    colors = ["#8d8d8d", "#4178a6", "#b65f45", "#6f6aa8", "#1f7a5c"]
    for i, (model, label, color) in enumerate(zip(models, labels, colors)):
        values = []
        for train_label, test_label in directions:
            row = df.loc[
                (df["train_dataset"] == train_label)
                & (df["test_dataset"] == test_label)
                & (df["model"] == model)
            ].iloc[0]
            values.append(row["mae"])
        plt.bar(x + (i - 2) * width, values, width=width, label=label, color=color)
    plt.xticks(x, [f"{a}\n→ {b}" for a, b in directions])
    plt.ylabel("External validation MAE (years)")
    plt.title("External MAE by model")
    plt.legend(frameon=False, ncols=3, fontsize=9)
    plt.grid(axis="y", alpha=0.18)
    savefig(figures_dir / "ensemble_vs_baseline_mae.png", dpi)


def figure_shap(results_dir: Path, figures_dir: Path, dpi: int) -> None:
    shap_df = pd.read_csv(results_dir / "shap_importance.csv").head(20).iloc[::-1]
    plt.figure(figsize=(7.5, 6.2))
    plt.barh(shap_df["cpg"], shap_df["mean_abs_shap_average"], color="#5a7d3a")
    plt.xlabel("Average mean absolute SHAP value (years)")
    plt.title("Top CpG drivers in gradient-boosting clock")
    plt.grid(axis="x", alpha=0.18)
    savefig(figures_dir / "shap_top_cpg_importance.png", dpi)


def figure_conformal(results_dir: Path, figures_dir: Path, dpi: int) -> None:
    conformal = load_json(results_dir / "conformal_results.json")["conformal_summary"]
    df = pd.DataFrame(conformal)
    df = df.loc[df["model"] == "elastic_net_cv"].copy()
    labels = [f"{r.train_dataset}\n→ {r.test_dataset}" for r in df.itertuples()]
    plt.figure(figsize=(7, 4.4))
    plt.bar(labels, df["empirical_coverage"], color="#6d7f9d", width=0.55)
    plt.axhline(0.90, color="#333333", linestyle="--", linewidth=1.1, label="Nominal 90%")
    plt.ylim(0, 1.05)
    plt.ylabel("Empirical coverage")
    plt.title("Split-conformal external interval coverage: ElasticNetCV")
    plt.legend(frameon=False)
    plt.grid(axis="y", alpha=0.18)
    savefig(figures_dir / "conformal_interval_coverage.png", dpi)


def figure_age_gap(results_dir: Path, figures_dir: Path, dpi: int) -> None:
    df = pd.read_csv(results_dir / "age_gap_no_sex_predictions.csv")
    df = df.loc[df["model"] == "elastic_net_cv_no_sex"].copy()
    directions = list(df.groupby(["train_dataset", "test_dataset"], sort=False).groups)
    plt.figure(figsize=(8, 4.6))
    colors = ["#2f6f73", "#b65f45"]
    for color, (train_label, test_label) in zip(colors, directions):
        frame = df.loc[(df["train_dataset"] == train_label) & (df["test_dataset"] == test_label)]
        plt.hist(
            frame["age_gap"],
            bins=28,
            alpha=0.48,
            density=True,
            color=color,
            label=f"{train_label} → {test_label}",
        )
    plt.axvline(0, color="#333333", linewidth=1.1, linestyle="--")
    plt.xlabel("Epigenetic age gap (predicted - chronological, years)")
    plt.ylabel("Density")
    plt.title("External-test age-gap distributions: no-sex ElasticNetCV")
    plt.legend(frameon=False)
    plt.grid(axis="y", alpha=0.18)
    savefig(figures_dir / "age_gap_distribution.png", dpi)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results", type=Path)
    parser.add_argument("--figures-dir", default="figures", type=Path)
    parser.add_argument("--dpi", default=300, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    figure_external_scatter(args.results_dir, args.figures_dir, args.dpi)
    figure_mae_bar(args.results_dir, args.figures_dir, args.dpi)
    figure_shap(args.results_dir, args.figures_dir, args.dpi)
    figure_conformal(args.results_dir, args.figures_dir, args.dpi)
    figure_age_gap(args.results_dir, args.figures_dir, args.dpi)


if __name__ == "__main__":
    main()
