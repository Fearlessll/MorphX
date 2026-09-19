"""Fit descriptive risk, clinical-only, and combined Cox models on matched rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from configs.cohorts import default_clinical_path, get_cohort
from prognosis.clinical import load_survival_labels, wsi_identifier


def prepare_table(predictions: pd.DataFrame, clinical: pd.DataFrame, cohort: str,
                  dataset: str, clinical_csv: Path, categorical: list[str]) -> pd.DataFrame:
    if not {"wsi_name", "risk_score"}.issubset(predictions.columns):
        raise ValueError("Prediction table requires wsi_name and risk_score")
    if predictions["wsi_name"].duplicated().any():
        raise ValueError("Prediction table contains duplicate WSI names")
    identifier = next((name for name in ("WSIs", "wsi_id", "WSI", "case_id", "patient_id",
                                         "submitter_id", "Sample ID") if name in clinical), None)
    if identifier is None:
        raise ValueError("Clinical table lacks a supported identifier column")
    missing_columns = sorted(set(categorical) - set(clinical.columns))
    if missing_columns:
        raise ValueError(f"Clinical table lacks categorical columns: {missing_columns}")
    names = predictions["wsi_name"].astype(str).tolist()
    events, times = load_survival_labels(names, cohort, dataset, clinical_csv=clinical_csv)
    frame = predictions.copy()
    frame["_id"] = [wsi_identifier(name, cohort, dataset) for name in names]
    frame["event"] = events.astype(int)
    frame["time"] = times
    frame["risk"] = pd.to_numeric(frame["risk_score"], errors="coerce")
    clinical = clinical.copy()
    clinical["_id"] = clinical[identifier].astype(str).str.strip()
    clinical = clinical.drop_duplicates("_id", keep="first")
    joined = frame.merge(clinical[["_id", *categorical]], on="_id", how="left", validate="many_to_one")
    if len(joined) != len(frame):
        raise RuntimeError("Clinical join changed the number of WSI rows")
    return joined


def fit_models(table: pd.DataFrame, categorical: list[str]) -> tuple[dict, pd.DataFrame]:
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index

    required = ["time", "event", "risk", *categorical]
    complete = table[required].replace([np.inf, -np.inf], np.nan).dropna()
    if len(complete) < 20 or complete["event"].sum() < 5:
        raise ValueError("At least 20 complete cases and 5 events are required")
    covariates = pd.get_dummies(complete[categorical].astype(str), drop_first=True,
                                dtype=float) if categorical else pd.DataFrame(index=complete.index)
    constant = [name for name in covariates if covariates[name].nunique() < 2]
    covariates = covariates.drop(columns=constant)
    if categorical and covariates.empty:
        raise ValueError("Categorical covariates have no variation after complete-case filtering")
    results = {}
    coefficients = []
    model_inputs = [("risk_only", complete[["risk"]])]
    if not covariates.empty:
        model_inputs.extend((("clinical_only", covariates),
                             ("combined", pd.concat([complete[["risk"]], covariates], axis=1))))
    for label, predictors in model_inputs:
        if predictors.empty:
            continue
        fit_frame = pd.concat([complete[["time", "event"]], predictors], axis=1)
        model = CoxPHFitter().fit(fit_frame, duration_col="time", event_col="event")
        predicted = model.predict_partial_hazard(fit_frame).to_numpy().reshape(-1)
        results[label] = {"n": len(fit_frame), "events": int(fit_frame["event"].sum()),
                          "cindex_in_sample": float(concordance_index(fit_frame["time"],
                                                                       -predicted, fit_frame["event"]))}
        for variable, row in model.summary.iterrows():
            coefficients.append({"model": label, "variable": variable,
                                 "hazard_ratio": float(row["exp(coef)"]),
                                 "ci95_low": float(row["exp(coef) lower 95%"]),
                                 "ci95_high": float(row["exp(coef) upper 95%"]),
                                 "p": float(row["p"])})
    summary = {"n_predictions": len(table), "n_complete": len(complete),
               "n_missing_or_invalid": len(table) - len(complete),
               "categorical_columns": categorical, "dropped_constant_dummies": constant,
               "models": results, "note": "Descriptive in-sample Cox fits; not an external validation estimate."}
    return summary, pd.DataFrame(coefficients)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Categorical clinical and MorphX-risk Cox analysis.")
    parser.add_argument("--cohort", choices=("hcc", "luad"), required=True)
    parser.add_argument("--dataset", choices=("tcga", "kmmufh", "gd"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument("--clinical-csv", type=Path)
    parser.add_argument("--categorical", nargs="*", default=[],
                        help="Clinical columns already categorized according to the study protocol.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    spec = get_cohort(args.cohort)
    if args.dataset not in spec.clinical_files:
        raise ValueError(f"{args.dataset} is not declared for {args.cohort}")
    clinical_csv = args.clinical_csv or default_clinical_path(args.data_root, spec, args.dataset)
    predictions = pd.read_csv(args.predictions_csv)
    clinical = pd.read_csv(clinical_csv)
    table = prepare_table(predictions, clinical, args.cohort, args.dataset,
                          clinical_csv, args.categorical)
    summary, coefficients = fit_models(table, args.categorical)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    coefficients.to_csv(args.output_dir / "coefficients.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
