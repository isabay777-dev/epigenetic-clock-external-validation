#!/usr/bin/env python3
"""Build and evaluate methylation-clock models on the GEO clock-CpG panels."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import warnings
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"sklearn\..*")

import numpy as np
import pandas as pd
import shap
from scipy import stats
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, VotingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_SEED = 20260703
DATASETS = {
    "GSE40279": "data/gse40279_clock_cpgs.csv",
    "GSE87571": "data/gse87571_clock_cpgs.csv",
}
METADATA_COLUMNS = ["dataset", "sample_geo_accession", "sample_title", "age", "sex", "tissue"]


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if math.isnan(float(value)):
            return None
        return float(value)
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if pd.isna(value) and not isinstance(value, str):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n")


def source_ref(obj: Any) -> str:
    lines, start = inspect.getsourcelines(obj)
    return f"make_clock.py:{start}-L{start + len(lines) - 1}"


def load_dataset(path: str, label: str, max_rows: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if max_rows is not None:
        df = df.head(max_rows).copy()
    df["dataset"] = label
    return df


def load_data(max_rows_per_dataset: int | None = None) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    frames = [
        load_dataset(path, label, max_rows=max_rows_per_dataset)
        for label, path in DATASETS.items()
    ]
    full = pd.concat(frames, ignore_index=True)
    cpg_cols = [c for c in full.columns if c not in METADATA_COLUMNS]
    audit: dict[str, Any] = {
        "cpg_feature_count": len(cpg_cols),
        "raw_rows": {},
        "usable_rows_after_missing_age_exclusion": {},
        "missing_beta_cells": {},
        "missing_age_cells": {},
        "missing_sex_cells": {},
        "imputation_strategy": (
            "CpG beta values are median-imputed using the training split only; "
            "sex is most-frequent imputed using the training split only. "
            "Rows missing chronological age are excluded because age is the supervised target."
        ),
    }
    for label, frame in full.groupby("dataset", sort=False):
        audit["raw_rows"][label] = int(len(frame))
        audit["usable_rows_after_missing_age_exclusion"][label] = int(frame["age"].notna().sum())
        audit["missing_beta_cells"][label] = int(frame[cpg_cols].isna().sum().sum())
        audit["missing_age_cells"][label] = int(frame["age"].isna().sum())
        audit["missing_sex_cells"][label] = int(frame["sex"].isna().sum())
    full = full.loc[full["age"].notna()].copy()
    full["age"] = full["age"].astype(float)
    audit["age_range_after_missing_age_exclusion"] = {}
    audit["sex_counts_after_missing_age_exclusion"] = {}
    for label, frame in full.groupby("dataset", sort=False):
        audit["age_range_after_missing_age_exclusion"][label] = {
            "min": float(frame["age"].min()),
            "max": float(frame["age"].max()),
        }
        audit["sex_counts_after_missing_age_exclusion"][label] = {
            str(k): int(v) for k, v in frame["sex"].fillna("").value_counts(dropna=False).to_dict().items()
        }
    return full, cpg_cols, audit


def feature_columns(cpg_cols: list[str], include_sex: bool) -> list[str]:
    return cpg_cols + (["sex"] if include_sex else [])


def make_preprocessor(cpg_cols: list[str], include_sex: bool) -> ColumnTransformer:
    transformers: list[tuple[str, Pipeline, list[str]]] = [
        (
            "cpg",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            cpg_cols,
        )
    ]
    if include_sex:
        transformers.append(
            (
                "sex",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                ["sex"],
            )
        )
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def make_pipeline(model: Any, cpg_cols: list[str], include_sex: bool) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", make_preprocessor(cpg_cols, include_sex)),
            ("model", model),
        ]
    )


def model_factories(cpg_cols: list[str], include_sex: bool, mlp_max_iter: int) -> dict[str, Any]:
    return {
        "elastic_net_cv": lambda: make_pipeline(
            ElasticNetCV(
                l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
                alphas=np.logspace(-2, 1.5, 30),
                cv=5,
                max_iter=20000,
                random_state=RANDOM_SEED,
                n_jobs=1,
            ),
            cpg_cols,
            include_sex,
        ),
        "hist_gradient_boosting": lambda: make_pipeline(
            HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=0.04,
                max_iter=350,
                max_leaf_nodes=15,
                l2_regularization=0.02,
                early_stopping=True,
                random_state=RANDOM_SEED,
            ),
            cpg_cols,
            include_sex,
        ),
        "mlp": lambda: make_pipeline(
            TransformedTargetRegressor(
                regressor=MLPRegressor(
                    hidden_layer_sizes=(64, 16),
                    activation="relu",
                    alpha=0.001,
                    learning_rate_init=0.001,
                    max_iter=mlp_max_iter,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=25,
                    random_state=RANDOM_SEED,
                ),
                transformer=StandardScaler(),
            ),
            cpg_cols,
            include_sex,
        ),
        "plain_elastic_net_baseline": lambda: make_pipeline(
            ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=20000, random_state=RANDOM_SEED),
            cpg_cols,
            include_sex,
        ),
    }


def make_ensemble(cpg_cols: list[str], include_sex: bool, mlp_max_iter: int) -> VotingRegressor:
    # A simple unweighted average avoids training a meta-model on these modest cohort sizes,
    # keeping external validation free of extra stacking-leakage choices.
    factories = model_factories(cpg_cols, include_sex, mlp_max_iter)
    return VotingRegressor(
        estimators=[
            ("elastic_net_cv", factories["elastic_net_cv"]()),
            ("hist_gradient_boosting", factories["hist_gradient_boosting"]()),
            ("mlp", factories["mlp"]()),
        ],
        n_jobs=None,
    )


def all_model_factories(cpg_cols: list[str], include_sex: bool, mlp_max_iter: int) -> dict[str, Any]:
    factories = model_factories(cpg_cols, include_sex, mlp_max_iter)
    factories["ensemble_average"] = lambda: make_ensemble(cpg_cols, include_sex, mlp_max_iter)
    return factories


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    pearson = stats.pearsonr(y_true, y_pred).statistic if len(np.unique(y_pred)) > 1 else np.nan
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
        "pearson_r": pearson,
        "median_absolute_error": float(np.median(np.abs(y_true - y_pred))),
        "n": int(len(y_true)),
    }


def run_external_validation(
    data: pd.DataFrame,
    cpg_cols: list[str],
    include_sex: bool,
    mlp_max_iter: int,
    output_dir: Path,
) -> tuple[dict[str, Any], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    features = feature_columns(cpg_cols, include_sex)
    factories = all_model_factories(cpg_cols, include_sex, mlp_max_iter)
    directions = [("GSE40279", "GSE87571"), ("GSE87571", "GSE40279")]
    for train_label, test_label in directions:
        train = data.loc[data["dataset"] == train_label].copy()
        test = data.loc[data["dataset"] == test_label].copy()
        x_train, y_train = train[features], train["age"].to_numpy()
        x_test, y_test = test[features], test["age"].to_numpy()
        for model_name, factory in factories.items():
            model = factory()
            model.fit(x_train, y_train)
            pred = model.predict(x_test)
            metrics = evaluate_predictions(y_test, pred)
            rows.append(
                {
                    "train_dataset": train_label,
                    "test_dataset": test_label,
                    "model": model_name,
                    **metrics,
                }
            )
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "train_dataset": train_label,
                        "test_dataset": test_label,
                        "model": model_name,
                        "sample_geo_accession": test["sample_geo_accession"].to_numpy(),
                        "chronological_age": y_test,
                        "predicted_age": pred,
                        "age_gap": pred - y_test,
                        "sex": test["sex"].to_numpy(),
                    }
                )
            )
    metrics_df = pd.DataFrame(rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(output_dir / "external_predictions.csv", index=False)
    return {"external_validation": rows, "provenance": source_ref(run_external_validation)}, predictions


def run_within_cohort_cv(
    data: pd.DataFrame,
    cpg_cols: list[str],
    include_sex: bool,
    mlp_max_iter: int,
    n_splits: int,
) -> dict[str, Any]:
    features = feature_columns(cpg_cols, include_sex)
    factories = all_model_factories(cpg_cols, include_sex, mlp_max_iter)
    fold_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for dataset_label, frame in data.groupby("dataset", sort=False):
        x = frame[features].reset_index(drop=True)
        y = frame["age"].to_numpy()
        actual_splits = min(n_splits, len(frame))
        splitter = KFold(n_splits=actual_splits, shuffle=True, random_state=RANDOM_SEED)
        for model_name, factory in factories.items():
            model_fold_metrics: list[dict[str, float]] = []
            for fold_index, (train_idx, test_idx) in enumerate(splitter.split(x), start=1):
                model = factory()
                model.fit(x.iloc[train_idx], y[train_idx])
                pred = model.predict(x.iloc[test_idx])
                metrics = evaluate_predictions(y[test_idx], pred)
                row = {
                    "dataset": dataset_label,
                    "model": model_name,
                    "fold": fold_index,
                    **metrics,
                }
                fold_rows.append(row)
                model_fold_metrics.append(metrics)
            metric_names = ["mae", "rmse", "pearson_r", "median_absolute_error"]
            summary = {"dataset": dataset_label, "model": model_name, "folds": actual_splits}
            for metric_name in metric_names:
                values = np.array([m[metric_name] for m in model_fold_metrics], dtype=float)
                summary[f"{metric_name}_mean"] = float(np.nanmean(values))
                summary[f"{metric_name}_sd"] = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
            summary_rows.append(summary)
    return {
        "within_cohort_cv_summary": summary_rows,
        "within_cohort_cv_folds": fold_rows,
        "provenance": source_ref(run_within_cohort_cv),
    }


def transformed_feature_names(model: Pipeline) -> list[str]:
    names = model.named_steps["preprocess"].get_feature_names_out()
    return [str(name) for name in names]


def compute_shap_importance(
    fitted_gb: Pipeline,
    x_background: pd.DataFrame,
    x_explain: pd.DataFrame,
    cpg_cols: list[str],
    max_samples: int,
) -> pd.Series:
    preprocessor = fitted_gb.named_steps["preprocess"]
    gb_model = fitted_gb.named_steps["model"]
    x_bg = preprocessor.transform(x_background)
    x_eval = preprocessor.transform(x_explain)
    feature_names = transformed_feature_names(fitted_gb)
    if len(x_eval) > max_samples:
        rng = np.random.default_rng(RANDOM_SEED)
        indices = np.sort(rng.choice(len(x_eval), size=max_samples, replace=False))
        x_eval = x_eval[indices]
    if len(x_bg) > max_samples:
        rng = np.random.default_rng(RANDOM_SEED + 1)
        indices = np.sort(rng.choice(len(x_bg), size=max_samples, replace=False))
        x_bg = x_bg[indices]
    try:
        explainer = shap.Explainer(gb_model, x_bg, feature_names=feature_names)
        values = explainer(x_eval)
    except Exception:
        explainer = shap.Explainer(gb_model.predict, x_bg, feature_names=feature_names, algorithm="permutation")
        values = explainer(x_eval, max_evals=max(2 * len(feature_names) + 1, 100))
    shap_values = np.asarray(values.values)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 0]
    mean_abs = pd.Series(np.abs(shap_values).mean(axis=0), index=feature_names)
    return mean_abs.loc[[c for c in cpg_cols if c in mean_abs.index]].sort_values(ascending=False)


def run_shap_analysis(
    data: pd.DataFrame,
    cpg_cols: list[str],
    include_sex: bool,
    mlp_max_iter: int,
    shap_sample_size: int,
    output_dir: Path,
) -> dict[str, Any]:
    features = feature_columns(cpg_cols, include_sex)
    directions = [("GSE40279", "GSE87571"), ("GSE87571", "GSE40279")]
    importances: dict[str, pd.Series] = {}
    for train_label, test_label in directions:
        train = data.loc[data["dataset"] == train_label].copy()
        test = data.loc[data["dataset"] == test_label].copy()
        gb = model_factories(cpg_cols, include_sex, mlp_max_iter)["hist_gradient_boosting"]()
        gb.fit(train[features], train["age"].to_numpy())
        key = f"train_{train_label}_test_{test_label}"
        importances[key] = compute_shap_importance(
            gb,
            train[features],
            test[features],
            cpg_cols,
            max_samples=shap_sample_size,
        )
    shap_df = pd.DataFrame({"cpg": cpg_cols})
    for key, series in importances.items():
        shap_df[f"mean_abs_shap_{key}"] = shap_df["cpg"].map(series)
        shap_df[f"rank_{key}"] = shap_df[f"mean_abs_shap_{key}"].rank(ascending=False, method="min")
        shap_df[f"top20_{key}"] = shap_df[f"rank_{key}"] <= 20
    first_key, second_key = list(importances)
    rank_corr = stats.spearmanr(
        shap_df[f"rank_{first_key}"],
        shap_df[f"rank_{second_key}"],
        nan_policy="omit",
    )
    top20_a = set(shap_df.loc[shap_df[f"top20_{first_key}"], "cpg"])
    top20_b = set(shap_df.loc[shap_df[f"top20_{second_key}"], "cpg"])
    shap_df["mean_abs_shap_average"] = shap_df[
        [f"mean_abs_shap_{first_key}", f"mean_abs_shap_{second_key}"]
    ].mean(axis=1)
    shap_df = shap_df.sort_values("mean_abs_shap_average", ascending=False)
    shap_df.to_csv(output_dir / "shap_importance.csv", index=False)
    return {
        "rank_stability": {
            "spearman_rho_all_418_cpgs": float(rank_corr.statistic),
            "spearman_p_value": float(rank_corr.pvalue),
            "top20_jaccard": len(top20_a & top20_b) / len(top20_a | top20_b),
            "top20_overlap_count": len(top20_a & top20_b),
            "direction_a": first_key,
            "direction_b": second_key,
        },
        "top20_by_average_mean_abs_shap": shap_df.head(20).to_dict(orient="records"),
        "provenance": source_ref(run_shap_analysis),
    }


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    q_level = min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, q_level, method="higher"))


def run_conformal(
    data: pd.DataFrame,
    cpg_cols: list[str],
    include_sex: bool,
    mlp_max_iter: int,
    calibration_fraction: float,
    alpha: float,
    output_dir: Path,
) -> tuple[dict[str, Any], pd.DataFrame]:
    features = feature_columns(cpg_cols, include_sex)
    directions = [("GSE40279", "GSE87571"), ("GSE87571", "GSE40279")]
    summary_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    conformal_models = {
        "elastic_net_cv": lambda: model_factories(cpg_cols, include_sex, mlp_max_iter)["elastic_net_cv"](),
        "ensemble_average": lambda: make_ensemble(cpg_cols, include_sex, mlp_max_iter),
    }
    for train_label, test_label in directions:
        train = data.loc[data["dataset"] == train_label].copy()
        test = data.loc[data["dataset"] == test_label].copy()
        proper, calibration = train_test_split(
            train,
            test_size=calibration_fraction,
            random_state=RANDOM_SEED,
            shuffle=True,
        )
        for model_name, factory in conformal_models.items():
            model = factory()
            model.fit(proper[features], proper["age"].to_numpy())
            calibration_pred = model.predict(calibration[features])
            scores = np.abs(calibration["age"].to_numpy() - calibration_pred)
            qhat = conformal_quantile(scores, alpha)
            test_pred = model.predict(test[features])
            lower = test_pred - qhat
            upper = test_pred + qhat
            y_test = test["age"].to_numpy()
            covered = (y_test >= lower) & (y_test <= upper)
            interval_width = upper - lower
            summary_rows.append(
                {
                    "train_dataset": train_label,
                    "test_dataset": test_label,
                    "model": model_name,
                    "role": "primary_recommended_model" if model_name == "elastic_net_cv" else "secondary_comparison",
                    "alpha": alpha,
                    "nominal_coverage": 1 - alpha,
                    "proper_training_n": int(len(proper)),
                    "calibration_n": int(len(calibration)),
                    "test_n": int(len(test)),
                    "qhat_absolute_error_years": qhat,
                    "empirical_coverage": float(np.mean(covered)),
                    "mean_interval_width_years": float(np.mean(interval_width)),
                    "median_interval_width_years": float(np.median(interval_width)),
                    **{f"point_{k}": v for k, v in evaluate_predictions(y_test, test_pred).items()},
                }
            )
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "train_dataset": train_label,
                        "test_dataset": test_label,
                        "model": model_name,
                        "role": "primary_recommended_model" if model_name == "elastic_net_cv" else "secondary_comparison",
                        "sample_geo_accession": test["sample_geo_accession"].to_numpy(),
                        "chronological_age": y_test,
                        "predicted_age": test_pred,
                        "age_gap": test_pred - y_test,
                        "interval_lower_90": lower,
                        "interval_upper_90": upper,
                        "interval_width": interval_width,
                        "covered": covered,
                        "sex": test["sex"].to_numpy(),
                    }
                )
            )
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(output_dir / "conformal_predictions.csv", index=False)
    return {
        "conformal_summary": summary_rows,
        "provenance": source_ref(run_conformal),
    }, predictions


def ols_coefficient_test(y: np.ndarray, x: np.ndarray, coefficient_index: int) -> dict[str, float]:
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    fitted = np.einsum("ij,j->i", x, beta)
    residual = y - fitted
    df = x.shape[0] - np.linalg.matrix_rank(x)
    sigma2 = float((residual @ residual) / df)
    cov = sigma2 * np.linalg.pinv(x.T @ x)
    se = np.sqrt(np.diag(cov))
    t_stat = float(beta[coefficient_index] / se[coefficient_index])
    p_value = float(2 * stats.t.sf(abs(t_stat), df=df))
    return {
        "coefficient": float(beta[coefficient_index]),
        "standard_error": float(se[coefficient_index]),
        "t_statistic": t_stat,
        "p_value": p_value,
        "df": int(df),
    }


def residualize(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    fitted = np.einsum("ij,j->i", x, beta)
    return y - fitted


def run_age_gap_analysis(
    data: pd.DataFrame,
    cpg_cols: list[str],
    output_dir: Path,
) -> tuple[dict[str, Any], pd.DataFrame]:
    features = feature_columns(cpg_cols, include_sex=False)
    directions = [("GSE40279", "GSE87571"), ("GSE87571", "GSE40279")]
    rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    for train_label, test_label in directions:
        train = data.loc[data["dataset"] == train_label].copy()
        test = data.loc[data["dataset"] == test_label].copy()
        model = model_factories(cpg_cols, include_sex=False, mlp_max_iter=1)["elastic_net_cv"]()
        model.fit(train[features], train["age"].to_numpy())
        pred = model.predict(test[features])
        frame = pd.DataFrame(
            {
                "train_dataset": train_label,
                "test_dataset": test_label,
                "model": "elastic_net_cv_no_sex",
                "sample_geo_accession": test["sample_geo_accession"].to_numpy(),
                "chronological_age": test["age"].to_numpy(),
                "predicted_age": pred,
                "age_gap": pred - test["age"].to_numpy(),
                "sex": test["sex"].to_numpy(),
            }
        )
        prediction_frames.append(frame)
        usable = frame.loc[frame["sex"].notna()].copy()
        usable["female_indicator"] = usable["sex"].str.lower().map({"female": 1.0, "male": 0.0})
        usable = usable.loc[usable["female_indicator"].notna()].copy()
        male_gap = usable.loc[usable["female_indicator"] == 0, "age_gap"]
        female_gap = usable.loc[usable["female_indicator"] == 1, "age_gap"]
        pearson = stats.pearsonr(usable["female_indicator"], usable["age_gap"])
        ttest = stats.ttest_ind(female_gap, male_gap, equal_var=False, nan_policy="omit")
        age_values = usable["chronological_age"].to_numpy(dtype=float)
        age_centered = age_values - float(age_values.mean())
        x_age_adjusted = np.column_stack(
            [
                np.ones(len(usable)),
                usable["female_indicator"].to_numpy(dtype=float),
                age_centered,
            ]
        )
        age_adjusted = ols_coefficient_test(
            usable["age_gap"].to_numpy(dtype=float),
            x_age_adjusted,
            coefficient_index=1,
        )
        x_age_only = np.column_stack([np.ones(len(usable)), age_centered])
        partial = stats.pearsonr(
            residualize(usable["female_indicator"].to_numpy(dtype=float), x_age_only),
            residualize(usable["age_gap"].to_numpy(dtype=float), x_age_only),
        )
        rows.append(
            {
                "train_dataset": train_label,
                "test_dataset": test_label,
                "model": "elastic_net_cv_no_sex",
                "sex_included_in_predictor": False,
                "n_with_nonmissing_binary_sex": int(len(usable)),
                "female_n": int((usable["female_indicator"] == 1).sum()),
                "male_n": int((usable["female_indicator"] == 0).sum()),
                "age_gap_mean_all": float(usable["age_gap"].mean()),
                "age_gap_sd_all": float(usable["age_gap"].std(ddof=1)),
                "age_gap_mean_female": float(female_gap.mean()),
                "age_gap_mean_male": float(male_gap.mean()),
                "female_minus_male_age_gap_years": float(female_gap.mean() - male_gap.mean()),
                "pearson_r_gap_vs_female_indicator": float(pearson.statistic),
                "pearson_p_value": float(pearson.pvalue),
                "welch_t_statistic_female_vs_male": float(ttest.statistic),
                "welch_t_p_value": float(ttest.pvalue),
                "age_adjusted_female_coefficient_years": age_adjusted["coefficient"],
                "age_adjusted_female_standard_error": age_adjusted["standard_error"],
                "age_adjusted_female_t_statistic": age_adjusted["t_statistic"],
                "age_adjusted_female_p_value": age_adjusted["p_value"],
                "age_adjusted_df": age_adjusted["df"],
                "age_adjusted_partial_r": float(partial.statistic),
                "age_adjusted_partial_r_p_value": float(partial.pvalue),
            }
        )
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(output_dir / "age_gap_no_sex_predictions.csv", index=False)
    return {"age_gap_sex_association": rows, "provenance": source_ref(run_age_gap_analysis)}, predictions


def run_model_comparison_bootstrap(
    external_predictions: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    linear_models = {"elastic_net_cv", "plain_elastic_net_baseline"}
    for (train_label, test_label), frame in external_predictions.groupby(["train_dataset", "test_dataset"], sort=False):
        pivot = frame.pivot_table(
            index="sample_geo_accession",
            columns="model",
            values=["chronological_age", "predicted_age"],
            aggfunc="first",
        )
        y = pivot["chronological_age"].iloc[:, 0].to_numpy(dtype=float)
        abs_errors = {
            model: np.abs(y - pivot["predicted_age"][model].to_numpy(dtype=float))
            for model in pivot["predicted_age"].columns
        }
        reference_model = min(linear_models, key=lambda model: float(abs_errors[model].mean()))
        ref_errors = abs_errors[reference_model]
        for model_name, model_errors in abs_errors.items():
            if model_name == reference_model:
                continue
            paired_diff = model_errors - ref_errors
            boot = np.empty(n_bootstrap, dtype=float)
            n = len(paired_diff)
            for i in range(n_bootstrap):
                idx = rng.integers(0, n, size=n)
                boot[i] = float(paired_diff[idx].mean())
            ci_low, ci_high = np.quantile(boot, [0.025, 0.975])
            rows.append(
                {
                    "train_dataset": train_label,
                    "test_dataset": test_label,
                    "reference_model": reference_model,
                    "comparison_model": str(model_name),
                    "difference_definition": "comparison_model_MAE_minus_reference_model_MAE",
                    "observed_mae_difference_years": float(paired_diff.mean()),
                    "ci_95_low": float(ci_low),
                    "ci_95_high": float(ci_high),
                    "bootstrap_resamples": int(n_bootstrap),
                    "seed": int(seed),
                    "test_n": int(n),
                    "ci_includes_zero": bool(ci_low <= 0 <= ci_high),
                }
            )
    return {
        "model_comparison_bootstrap": rows,
        "provenance": source_ref(run_model_comparison_bootstrap),
    }


def make_facts_markdown(
    output_dir: Path,
    audit: dict[str, Any],
    external: dict[str, Any],
    cv: dict[str, Any],
    shap_results: dict[str, Any],
    conformal: dict[str, Any],
    age_gap: dict[str, Any],
    bootstrap: dict[str, Any],
) -> str:
    lines: list[str] = [
        "# Article 3 Methylation Clock Analysis Facts",
        "",
        "Generated by `make_clock.py` from the two pre-existing GEO-derived CSV files in `data/`.",
        "No simulated, synthetic, or fabricated data are used.",
        "",
        "## Data Audit",
        "",
        f"- CpG feature count: {audit['cpg_feature_count']} (`{source_ref(load_data)}`).",
        f"- Imputation strategy: {audit['imputation_strategy']}",
        "",
        "| Dataset | Raw rows | Usable rows after missing-age exclusion | Missing beta cells | Missing age cells | Missing sex cells | Provenance |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for dataset in DATASETS:
        lines.append(
            f"| {dataset} | {audit['raw_rows'][dataset]} | "
            f"{audit['usable_rows_after_missing_age_exclusion'][dataset]} | "
            f"{audit['missing_beta_cells'][dataset]} | {audit['missing_age_cells'][dataset]} | "
            f"{audit['missing_sex_cells'][dataset]} | `{source_ref(load_data)}` |"
        )
    lines.extend(
        [
            "",
            "| Dataset | Age range after exclusion | Sex composition after exclusion |",
            "|---|---:|---|",
        ]
    )
    for dataset in DATASETS:
        age_range = audit["age_range_after_missing_age_exclusion"][dataset]
        sex_counts = audit["sex_counts_after_missing_age_exclusion"][dataset]
        sex_text = ", ".join(f"{sex or 'missing'}: {count}" for sex, count in sex_counts.items())
        lines.append(
            f"| {dataset} | {age_range['min']:.1f}-{age_range['max']:.1f} | {sex_text} |"
        )
    lines.extend(
        [
            "",
            "## External Validation",
            "",
            f"Metrics are computed in `{source_ref(evaluate_predictions)}` and fit/prediction rows are produced in `{external['provenance']}`.",
            "",
            "| Train | Test | Model | n | MAE | RMSE | Pearson r | Median absolute error |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in external["external_validation"]:
        lines.append(
            f"| {row['train_dataset']} | {row['test_dataset']} | {row['model']} | {row['n']} | "
            f"{row['mae']:.6f} | {row['rmse']:.6f} | {row['pearson_r']:.6f} | "
            f"{row['median_absolute_error']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Paired Bootstrap Model Comparison",
            "",
            f"MAE-difference confidence intervals are produced in `{bootstrap['provenance']}`. Positive differences mean the comparison model has larger MAE than the best linear reference model for that transfer direction.",
            "",
            "| Train | Test | Reference linear model | Comparison model | MAE difference | 95% CI | Includes zero |",
            "|---|---|---|---|---:|---:|---|",
        ]
    )
    for row in bootstrap["model_comparison_bootstrap"]:
        lines.append(
            f"| {row['train_dataset']} | {row['test_dataset']} | {row['reference_model']} | "
            f"{row['comparison_model']} | {row['observed_mae_difference_years']:.6f} | "
            f"{row['ci_95_low']:.6f} to {row['ci_95_high']:.6f} | {row['ci_includes_zero']} |"
        )
    lines.extend(
        [
            "",
            "## Within-Cohort 10-Fold CV",
            "",
            f"Cross-validation rows are produced in `{cv['provenance']}`.",
            "",
            "| Dataset | Model | Folds | MAE mean | MAE SD | RMSE mean | Pearson r mean | Median AE mean |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in cv["within_cohort_cv_summary"]:
        lines.append(
            f"| {row['dataset']} | {row['model']} | {row['folds']} | "
            f"{row['mae_mean']:.6f} | {row['mae_sd']:.6f} | {row['rmse_mean']:.6f} | "
            f"{row['pearson_r_mean']:.6f} | {row['median_absolute_error_mean']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## SHAP Stability",
            "",
            f"SHAP importances are produced in `{shap_results['provenance']}` and saved to `{output_dir / 'shap_importance.csv'}`.",
            "",
        ]
    )
    stability = shap_results["rank_stability"]
    for key, value in stability.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "| Rank | CpG | Average mean absolute SHAP |", "|---:|---|---:|"])
    for i, row in enumerate(shap_results["top20_by_average_mean_abs_shap"][:20], start=1):
        lines.append(f"| {i} | {row['cpg']} | {row['mean_abs_shap_average']:.6f} |")
    lines.extend(
        [
            "",
            "## Split-Conformal Prediction",
            "",
            f"Conformal intervals are produced in `{conformal['provenance']}` using the finite-sample quantile in `{source_ref(conformal_quantile)}`.",
            "",
            "| Train | Test | Model | Role | Calibration n | Test n | qhat | Empirical coverage | Mean width | Point MAE | Point RMSE | Point r |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in conformal["conformal_summary"]:
        lines.append(
            f"| {row['train_dataset']} | {row['test_dataset']} | {row['model']} | {row['role']} | "
            f"{row['calibration_n']} | {row['test_n']} | "
            f"{row['qhat_absolute_error_years']:.6f} | {row['empirical_coverage']:.6f} | "
            f"{row['mean_interval_width_years']:.6f} | {row['point_mae']:.6f} | "
            f"{row['point_rmse']:.6f} | {row['point_pearson_r']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Age-Gap Sex Association",
            "",
            f"Age-gap tests are produced in `{age_gap['provenance']}` from external ElasticNetCV predictions trained without sex as a feature.",
            "",
            "| Train | Test | n | Female n | Male n | Mean gap | Female - male | Raw r | Raw p | Age-adjusted female coefficient | Age-adjusted p | Partial r |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in age_gap["age_gap_sex_association"]:
        lines.append(
            f"| {row['train_dataset']} | {row['test_dataset']} | {row['n_with_nonmissing_binary_sex']} | "
            f"{row['female_n']} | {row['male_n']} | {row['age_gap_mean_all']:.6f} | "
            f"{row['female_minus_male_age_gap_years']:.6f} | "
            f"{row['pearson_r_gap_vs_female_indicator']:.6f} | {row['pearson_p_value']:.6g} | "
            f"{row['age_adjusted_female_coefficient_years']:.6f} | {row['age_adjusted_female_p_value']:.6g} | "
            f"{row['age_adjusted_partial_r']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            f"- `{output_dir / 'metrics.json'}`",
            f"- `{output_dir / 'external_validation.json'}`",
            f"- `{output_dir / 'external_predictions.csv'}`",
            f"- `{output_dir / 'shap_results.json'}`",
            f"- `{output_dir / 'shap_importance.csv'}`",
            f"- `{output_dir / 'conformal_results.json'}`",
            f"- `{output_dir / 'conformal_predictions.csv'}`",
            f"- `{output_dir / 'age_gap_results.json'}`",
            f"- `{output_dir / 'age_gap_no_sex_predictions.csv'}`",
            f"- `{output_dir / 'model_comparison_bootstrap.json'}`",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results", type=Path)
    parser.add_argument("--facts-file", default="ARTICLE3_methylation_facts.md", type=Path)
    parser.add_argument("--max-rows-per-dataset", default=None, type=int)
    parser.add_argument("--cv-folds", default=10, type=int)
    parser.add_argument("--include-sex", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mlp-max-iter", default=500, type=int)
    parser.add_argument("--shap-sample-size", default=200, type=int)
    parser.add_argument("--conformal-calibration-fraction", default=0.2, type=float)
    parser.add_argument("--alpha", default=0.10, type=float)
    parser.add_argument("--bootstrap-resamples", default=1000, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data, cpg_cols, audit = load_data(args.max_rows_per_dataset)
    external, external_predictions = run_external_validation(
        data, cpg_cols, args.include_sex, args.mlp_max_iter, args.output_dir
    )
    cv = run_within_cohort_cv(data, cpg_cols, args.include_sex, args.mlp_max_iter, args.cv_folds)
    shap_results = run_shap_analysis(
        data, cpg_cols, args.include_sex, args.mlp_max_iter, args.shap_sample_size, args.output_dir
    )
    conformal, _ = run_conformal(
        data,
        cpg_cols,
        args.include_sex,
        args.mlp_max_iter,
        args.conformal_calibration_fraction,
        args.alpha,
        args.output_dir,
    )
    age_gap, _ = run_age_gap_analysis(data, cpg_cols, args.output_dir)
    bootstrap = run_model_comparison_bootstrap(
        external_predictions,
        n_bootstrap=args.bootstrap_resamples,
        seed=RANDOM_SEED,
    )
    metrics = {"data_audit": audit, **cv}
    write_json(args.output_dir / "metrics.json", metrics)
    write_json(args.output_dir / "external_validation.json", external)
    write_json(args.output_dir / "shap_results.json", shap_results)
    write_json(args.output_dir / "conformal_results.json", conformal)
    write_json(args.output_dir / "age_gap_results.json", age_gap)
    write_json(args.output_dir / "model_comparison_bootstrap.json", bootstrap)
    facts = make_facts_markdown(
        args.output_dir,
        audit,
        external,
        cv,
        shap_results,
        conformal,
        age_gap,
        bootstrap,
    )
    args.facts_file.write_text(facts + "\n")


if __name__ == "__main__":
    main()
