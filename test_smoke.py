#!/usr/bin/env python3
"""Smoke test for the methylation-clock pipeline on a small real-data subset."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def test_pipeline_smoke() -> None:
    with tempfile.TemporaryDirectory(prefix="article3_smoke_") as tmp:
        tmp_path = Path(tmp)
        results_dir = tmp_path / "results"
        figures_dir = tmp_path / "figures"
        facts_file = tmp_path / "facts.md"
        run(
            [
                sys.executable,
                "make_clock.py",
                "--output-dir",
                str(results_dir),
                "--facts-file",
                str(facts_file),
                "--max-rows-per-dataset",
                "48",
                "--cv-folds",
                "2",
                "--mlp-max-iter",
                "80",
                "--shap-sample-size",
                "12",
            ]
        )
        run(
            [
                sys.executable,
                "make_figures.py",
                "--results-dir",
                str(results_dir),
                "--figures-dir",
                str(figures_dir),
                "--dpi",
                "120",
            ]
        )
        required = [
            results_dir / "metrics.json",
            results_dir / "external_validation.json",
            results_dir / "external_predictions.csv",
            results_dir / "shap_results.json",
            results_dir / "shap_importance.csv",
            results_dir / "conformal_results.json",
            results_dir / "conformal_predictions.csv",
            results_dir / "age_gap_results.json",
            facts_file,
            figures_dir / "external_predicted_vs_chronological.png",
            figures_dir / "ensemble_vs_baseline_mae.png",
            figures_dir / "shap_top_cpg_importance.png",
            figures_dir / "conformal_interval_coverage.png",
            figures_dir / "age_gap_distribution.png",
        ]
        for path in required:
            assert path.exists(), path
            assert path.stat().st_size > 0, path
        external = json.loads((results_dir / "external_validation.json").read_text())
        assert len(external["external_validation"]) == 10


if __name__ == "__main__":
    test_pipeline_smoke()
