"""Rank morphological attributions and transfer discovery cutoffs to external data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from configs.cohorts import get_cohort
from prognosis.clinical import load_survival_labels


def top_feature_frequency(scores: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    needed = {"wsi_id", "scope", "feature_name", "attribution_sum"}
    if not needed.issubset(scores.columns):
        raise ValueError(f"Feature scores lack {sorted(needed - set(scores.columns))}")
    ranked = scores.sort_values(["wsi_id", "scope", "attribution_sum"],
                                ascending=[True, True, False])
    ranked = ranked.groupby(["wsi_id", "scope"], sort=False).head(top_n)
    frequency = (ranked.groupby(["scope", "feature_name"]).size()
                 .rename("top_n_count").reset_index())
    return frequency.sort_values(["scope", "top_n_count", "feature_name"],
                                 ascending=[True, False, True]).reset_index(drop=True)


def _endpoint_table(scores: pd.DataFrame, cohort: str, dataset: str,
                    data_root: Path, clinical_csv: Path | None, scope: str,
                    feature_name: str) -> pd.DataFrame:
    selected = scores.loc[(scores["scope"] == scope) & (scores["feature_name"] == feature_name)].copy()
    if selected.empty or selected["wsi_id"].duplicated().any():
        raise ValueError(f"Missing or duplicated feature {scope}/{feature_name} in {dataset}")
    events, times = load_survival_labels(selected["wsi_id"], cohort, dataset,
                                         data_root=data_root, clinical_csv=clinical_csv)
    selected["event"] = events.astype(int)
    selected["time"] = times
    selected["value"] = pd.to_numeric(selected["feature_mean"], errors="coerce")
    if cohort == "luad" and scope == "nec":
        selected.loc[selected["n_pixels"] == 0, "value"] = 0.0
    selected = selected.replace([np.inf, -np.inf], np.nan).dropna(subset=["value", "time", "event"])
    if len(selected) < 20:
        raise ValueError(f"Only {len(selected)} usable feature rows for {dataset}")
    return selected


def _stats(table: pd.DataFrame, cutoff: float) -> dict:
    from lifelines import CoxPHFitter
    from lifelines.statistics import logrank_test

    high = table["value"] > cutoff
    if high.nunique() != 2:
        raise ValueError("Cutoff creates only one risk group")
    left, right = table.loc[~high], table.loc[high]
    logrank = logrank_test(left["time"], right["time"], left["event"], right["event"])
    model_frame = pd.DataFrame({"time": table["time"], "event": table["event"],
                                "high": high.astype(int)})
    cox = CoxPHFitter().fit(model_frame, "time", "event")
    coefficient = float(cox.params_["high"])
    interval = cox.confidence_intervals_.loc["high"]
    return {"n": len(table), "events": int(table["event"].sum()),
            "high_n": int(high.sum()), "low_n": int((~high).sum()),
            "logrank_p": float(logrank.p_value), "hazard_ratio": float(np.exp(coefficient)),
            "hr_ci95": [float(np.exp(interval.iloc[0])), float(np.exp(interval.iloc[1]))]}


def discovery_cutoff(table: pd.DataFrame, min_fraction: float = 0.1) -> tuple[float, dict]:
    values = table["value"].to_numpy(dtype=float)
    low, high = np.quantile(values, [0.1, 0.9])
    candidates = np.unique(values[(values >= low) & (values <= high)])
    best = None
    for cutoff in candidates:
        high_count = int((values > cutoff).sum())
        if min(high_count, len(values) - high_count) < min_fraction * len(values):
            continue
        try:
            result = _stats(table, float(cutoff))
        except Exception:
            continue
        if best is None or result["logrank_p"] < best[1]["logrank_p"]:
            best = (float(cutoff), result)
    if best is None:
        raise ValueError("No estimable discovery cutoff within the 10th–90th percentile interval")
    return best


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="MorphX feature-level attribution and biomarker analysis.")
    sub = parser.add_subparsers(dest="command", required=True)
    rank = sub.add_parser("rank", help="Count each feature's top-10 attribution frequency.")
    rank.add_argument("--scores-csv", type=Path, required=True)
    rank.add_argument("--output-csv", type=Path, required=True)
    rank.add_argument("--top-n", type=int, default=10)
    validate = sub.add_parser("validate", help="Fit cutoff in discovery and transfer unchanged to external data.")
    validate.add_argument("--cohort", choices=("hcc", "luad"), required=True)
    validate.add_argument("--data-root", type=Path, required=True)
    validate.add_argument("--discovery-scores-csv", type=Path, required=True)
    validate.add_argument("--external-scores-csv", type=Path, required=True)
    validate.add_argument("--discovery-clinical-csv", type=Path)
    validate.add_argument("--external-clinical-csv", type=Path)
    validate.add_argument("--scope", required=True)
    validate.add_argument("--feature-name", required=True)
    validate.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "rank":
        result = top_feature_frequency(pd.read_csv(args.scores_csv), args.top_n)
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output_csv, index=False)
        print(f"Ranked {result['feature_name'].nunique()} features")
        return
    spec = get_cohort(args.cohort)
    discovery = _endpoint_table(pd.read_csv(args.discovery_scores_csv), args.cohort,
                                spec.discovery_dataset, args.data_root,
                                args.discovery_clinical_csv, args.scope, args.feature_name)
    external = _endpoint_table(pd.read_csv(args.external_scores_csv), args.cohort,
                               spec.external_dataset, args.data_root,
                               args.external_clinical_csv, args.scope, args.feature_name)
    cutoff, discovery_result = discovery_cutoff(discovery)
    external_result = _stats(external, cutoff)
    report = {"cohort": args.cohort, "scope": args.scope, "feature_name": args.feature_name,
              "discovery_cutoff": cutoff, "cutoff_rule": "max-logrank in discovery, 10th–90th percentile",
              "discovery": discovery_result, "external": external_result}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
