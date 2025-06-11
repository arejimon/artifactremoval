# utils_runs.py
# High-level helpers to
#   1.  log / save a trained model with its hyper-parameters
#   2.  create publication-quality hyper-parameter-tuning figures
# Author: 2025-04-23

from __future__ import annotations
from pathlib import Path
from datetime import datetime
from typing import List, Type

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import keras_tuner as kt


# ───────────────────────────────────────────────────────────────
# 1.  Internal helpers for path / index management
# ───────────────────────────────────────────────────────────────
def _flatten_hps(hps) -> dict:
    """Return an ordered dict of HPs from Keras-Tuner or plain dict."""
    items = hps.values.items() if hasattr(hps, "values") else hps.items()
    return {k: float(v) if isinstance(v, (int, float)) else v for k, v in sorted(items)}


def _get_paths(
    experiment: str,
    best_hps,
    base_dir: str = "runs",
    model_ext: str = ".h5",
):
    """
    Ensure the run folder & index exist and return:
        model_path – timestamped file name for model saving
        fig_dir    – folder for figures
        row        – pandas Series representing this HP config
    """
    exp_dir = Path(base_dir) / experiment
    exp_dir.mkdir(parents=True, exist_ok=True)
    index_csv = exp_dir / "config_index.csv"

    df = pd.read_csv(index_csv, index_col="ID") if index_csv.exists() else pd.DataFrame()
    df.index.name = "ID"

    hp_dict = _flatten_hps(best_hps)

    if df.empty:
        cfg_id = "001"
        df = pd.DataFrame([{**hp_dict, "timestamp": datetime.now().isoformat(timespec="seconds")}],
                          index=[cfg_id])
    else:
        mask = (
            df.drop(columns=["timestamp", "notes"], errors="ignore")
            == pd.Series(hp_dict)
        ).all(axis=1)
        cfg_id = mask.idxmax() if mask.any() else f"{len(df) + 1:03d}"
        if not mask.any():
            df.loc[cfg_id] = {**hp_dict, "timestamp": datetime.now().isoformat(timespec="seconds")}

    df.to_csv(index_csv)

    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    model_path = exp_dir / f"{experiment}_ID{cfg_id}_{ts}{model_ext}"
    fig_dir = exp_dir / f"fig_ID{cfg_id}_{ts}"
    fig_dir.mkdir(exist_ok=True)

    return model_path, fig_dir, df.loc[cfg_id]


# ───────────────────────────────────────────────────────────────
# 2.  Public API – save a trained model
# ───────────────────────────────────────────────────────────────
def save_model(model, experiment: str, best_hps, base_dir: str = "runs") -> Path:
    """
    Save a compiled/fit Keras model with a timestamped name and
    record its hyper-parameters in runs/<experiment>/config_index.csv.

    Returns
    -------
    Path – full path where the model was saved
    """
    model_path, _, _ = _get_paths(experiment, best_hps, base_dir)
    model.save(model_path)
    return model_path


# ───────────────────────────────────────────────────────────────
# 3.  Internal helpers for summary plots
# ───────────────────────────────────────────────────────────────
def _prep_bins(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    df2["lr_bin"] = pd.cut(
        np.log10(df2["learning_rate"]),
        bins=[-6, -5, -4.3, -3.7, -3, -2.3],
        labels=["1e-6–5e-6", "5e-6–1e-4", "1e-4–5e-4",
                "5e-4–1e-3", "1e-3–5e-3"],
    )
    return df2


# ───────── internal summary-plot helper (no FutureWarnings) ──────────
def _create_hp_summary_figures(df_raw: pd.DataFrame, metric: str, out_dir: Path):
    """
    Produce:
      01_bar_summary.pdf      – mean±sd bar chart per HP value
      02_strip_grid.pdf       – small-multiple strip plots
      03_heat_lr_drop1.pdf    – LR×dropout1 heat-map
      04_heat_lr_batch.pdf    – LR×batch heat-map
    """
    out_dir.mkdir(exist_ok=True)
    df = _prep_bins(df_raw)
    hp_cols = ["lr_bin", "dropout_rate1", "dropout_rate2",
               "batch_size", "dense_units"]

    # A. Bar chart (mean ± SD per categorical value)
    agg_rows = []
    for hp in hp_cols:
        tmp = (
            df.groupby(hp, observed=False)[metric]   # ← observed=False
              .agg(["mean", "std"])
              .reset_index()
        )
        tmp["hp"] = hp
        agg_rows.append(tmp)
    bar_df = pd.concat(agg_rows, ignore_index=True)

    plt.figure(figsize=(9, 4))
    sns.barplot(data=bar_df, x="hp", y="mean",
                hue=bar_df.iloc[:, 0], dodge=False)
    plt.ylabel(f"Mean {metric}")
    plt.title("Average performance per hyper-parameter value")
    plt.legend(title="", bbox_to_anchor=(1.02, 1))
    plt.tight_layout()
    plt.savefig(out_dir / "01_bar_summary.pdf")
    plt.close()

    # B. Strip-grid of trial points
    long = df.melt(id_vars=[metric], value_vars=hp_cols,
                   var_name="hp", value_name="val")
    g = sns.catplot(data=long, x="val", y=metric,
                    col="hp", kind="strip", jitter=0.25,
                    height=3, col_wrap=3, sharey=False)
    g.set_titles("{col_name}"); g.set_xlabels("")
    g.fig.suptitle("Trial accuracy by hyper-parameter", y=1.02)
    g.fig.tight_layout()
    g.fig.savefig(out_dir / "02_strip_grid.pdf")
    plt.close(g.fig)

    # C. Two interaction heat-maps
    pivot = lambda x, y: df.pivot_table(
        metric, x, y, aggfunc="median", observed=False   # ← observed=False
    )

    for x, y, fname in [
        ("lr_bin", "dropout_rate1", "03_heat_lr_drop1"),
        ("lr_bin", "batch_size",    "04_heat_lr_batch"),
    ]:
        plt.figure(figsize=(5.2, 3.8))
        sns.heatmap(pivot(x, y), annot=True, fmt=".3f",
                    cmap="viridis", cbar_kws={"label": f"median {metric}"})
        plt.xlabel(x); plt.ylabel(y)
        plt.tight_layout()
        plt.savefig(out_dir / f"{fname}.pdf")
        plt.close()

# ───────────────────────────────────────────────────────────────
# 4.  Public API – plot tuner results
# ───────────────────────────────────────────────────────────────
def plot_tuner_results(
    build_model_fn,
    proj_dir: str,
    proj_name: str,
    experiment: str,
    base_dir: str = "runs",
    tuner_cls: Type[kt.Tuner] = kt.RandomSearch,
    metric: str = "val_accuracy",
    hp_keys: List[str] | None = None,
) -> pd.DataFrame:
    """
    Reload a completed Keras-Tuner project and drop summary figures
    into the matching runs/<experiment>/fig_* folder.

    Returns
    -------
    DataFrame – one row per trial with selected HPs and metric
    """
    hp_keys = hp_keys or ["learning_rate", "dropout_rate1", "dropout_rate2",
                          "batch_size", "dense_units"]

    tuner = tuner_cls(build_model_fn, objective=metric,
                      max_trials=1, directory=proj_dir, project_name=proj_name)
    tuner.reload()

    records = []
    for tr in tuner.oracle.trials.values():
        hp = tr.hyperparameters.values
        row = {"trial_id": int(tr.trial_id),
               metric: tr.metrics.get_last_value(metric)}
        row.update({k: hp.get(k) for k in hp_keys})
        records.append(row)
    df = pd.DataFrame(records).sort_values("trial_id")

    # Make figure directory aligned with this experiment & HP set
    _, fig_dir, _ = _get_paths(experiment, tuner.oracle.hyperparameters, base_dir)
    _create_hp_summary_figures(df, metric, fig_dir)

    return df
