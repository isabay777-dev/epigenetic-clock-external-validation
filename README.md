# Epigenetic Clock External Validation Study

This repository contains the analysis code, reduced GEO-derived methylation data, results, figures, and submission package for the manuscript:

**External validation, conformal uncertainty, and feature stability in machine-learning epigenetic clocks trained on established methylation panels**

The study is a reality-check analysis of chronological-age prediction on a fixed 418-CpG union of established Hannum and Horvath methylation-clock probes. It compares regularized linear models with more complex machine-learning models under bidirectional external validation, evaluates split-conformal prediction intervals, examines SHAP feature-stability, and summarizes exploratory no-sex age-gap sex associations. The main modeling result is conservative: on this established CpG panel and these two cohorts, regularized linear models matched or outperformed more complex models in external validation.

## Data Sources

The reduced CSV files in `data/` are derived from public Gene Expression Omnibus data:

- GSE40279
- GSE87571

The clock-CpG lists are from bio-learn v0.9.1 (`Hannum.csv` and `Horvath1.csv`). See `README_data.md` and `data/facts.md` for source URLs, retrieval notes, sample counts, missing-data notes, and CpG matching details.

## Reproduce

Install the listed Python dependencies, then run from the project root:

```bash
python3 fetch_methylation_data.py
python3 make_clock.py
python3 make_figures.py
```

`fetch_methylation_data.py` rebuilds the reduced GEO-derived clock-CpG CSV files. `make_clock.py` writes analysis outputs to `results/` and updates `ARTICLE3_methylation_facts.md`. `make_figures.py` writes manuscript figures to `figures/`.

## File Manifest

- `ARTICLE3_methylation_manuscript.md`: finalized manuscript text.
- `ARTICLE3_methylation_facts.md`: generated audit facts and result summaries used by the manuscript.
- `fetch_methylation_data.py`: downloads and reduces GEO methylation data to established clock CpGs.
- `make_clock.py`: runs preprocessing, model fitting, external validation, bootstrap comparison, conformal intervals, SHAP summaries, and age-gap analyses.
- `make_figures.py`: creates manuscript figures from files in `results/`.
- `requirements.txt`: Python dependencies used by the data analysis and plotting scripts.
- `test_smoke.py`: lightweight reproducibility smoke test.
- `data/`: reduced GEO-derived clock-CpG CSV files and data audit notes.
- `results/`: generated numerical outputs used by the manuscript.
- `figures/`: generated manuscript figures.
- `submission/`: title page, blinded manuscript, cover letter, and submission checklist.

## License

The repository code is released under the MIT License. See `LICENSE`.

The public GEO source data remain subject to their original database records and source terms. Manuscript text, figures, and third-party source data are not relicensed by this software license.
