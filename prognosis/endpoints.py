"""Reproducible secondary endpoint analyses for MorphX.

These helpers operate on tables and never train or tune a model. This keeps
secondary endpoint analyses separate from discovery-cohort model fitting and
locked external validation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from prognosis.utils import cindex_lifeline


EARLY_DEATH = "Death_within_36_months"
SURVIVED_BEYOND = "Survived_beyond_36_months"
EXCLUDED_CENSORED = "Excluded_censored_before_36_months"
EXCLUDED_INVALID = "Excluded_invalid_endpoint"


def _event_values(values: Iterable[object], column: str) -> pd.Series:
    """Normalize numeric and common text event encodings to nullable integers."""

    series = pd.Series(values)
    numeric = pd.to_numeric(series, errors="coerce")
    missing = numeric.isna()
    if missing.any():
        normalized = series.astype("string").str.strip().str.lower()
        mapped = normalized.map(
            {
                "1": 1,
                "0": 0,
                "true": 1,
                "false": 0,
                "yes": 1,
                "no": 0,
                "dead": 1,
                "deceased": 1,
                "event": 1,
                "alive": 0,
                "living": 0,
                "censored": 0,
                "censor": 0,
            }
        )
        numeric = numeric.where(~missing, mapped)
    numeric = pd.to_numeric(numeric, errors="coerce")
    valid = numeric.isin([0.0, 1.0])
    result = numeric.where(valid).astype("Int64")
    result.index = series.index
    return result.rename(column)


def assign_three_year_os_groups(
    frame: pd.DataFrame,
    *,
    time_col: str = "os_time_months",
    event_col: str = "os_status",
    threshold_months: float = 36.0,
    output_col: str = "os_3year_subgroup",
) -> pd.Series:
    """Apply the prespecified three-year OS grouping rule."""

    if threshold_months <= 0:
        raise ValueError("threshold_months must be positive.")
    if time_col not in frame.columns or event_col not in frame.columns:
        missing = [name for name in (time_col, event_col) if name not in frame.columns]
        raise KeyError(f"Missing required OS endpoint columns: {missing}")

    times = pd.to_numeric(frame[time_col], errors="coerce")
    events = _event_values(frame[event_col], event_col)
    valid_time = times.notna() & np.isfinite(times) & (times > 0)
    valid_event = events.notna()
    labels = pd.Series(EXCLUDED_INVALID, index=frame.index, dtype="string", name=output_col)
    early_death = valid_time & valid_event & (events == 1) & (times <= threshold_months)
    survived_beyond = valid_time & valid_event & (times > threshold_months)
    censored_before = valid_time & valid_event & (events == 0) & (times <= threshold_months)
    labels.loc[early_death] = EARLY_DEATH
    labels.loc[survived_beyond] = SURVIVED_BEYOND
    labels.loc[censored_before] = EXCLUDED_CENSORED
    return labels


def _safe_zscore(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    standard_deviation = numeric.std(ddof=0)
    if not np.isfinite(standard_deviation) or standard_deviation == 0:
        return pd.Series(np.nan, index=values.index, dtype=float)
    return (numeric - numeric.mean()) / standard_deviation


def _empty_cox_result(n: int, events: int, error: str = "") -> dict:
    return {
        "n": int(n),
        "events": int(events),
        "hr_per_sd": float("nan"),
        "ci95_low": float("nan"),
        "ci95_high": float("nan"),
        "p": float("nan"),
        "status": "not_fit",
        "error": error,
    }


def _fit_cox_per_sd(frame: pd.DataFrame, time_col: str, event_col: str, score_col: str) -> dict:
    """Fit a one-variable Cox model without changing the risk score."""

    data = frame[[time_col, event_col, score_col]].replace([np.inf, -np.inf], np.nan).dropna()
    events = int(pd.to_numeric(data[event_col], errors="coerce").sum()) if len(data) else 0
    if len(data) < 20 or events < 5 or data[score_col].nunique() < 2:
        return _empty_cox_result(
            len(data), events, "Insufficient complete cases, events, or score variation."
        )

    try:
        from lifelines import CoxPHFitter

        model_data = data.rename(
            columns={time_col: "duration", event_col: "event", score_col: "score"}
        )
        model = CoxPHFitter()
        model.fit(model_data[["duration", "event", "score"]], "duration", "event")
        coefficient = float(model.params_["score"])
        confidence = model.confidence_intervals_.loc["score"]
        return {
            "n": int(len(data)),
            "events": events,
            "hr_per_sd": float(np.exp(coefficient)),
            "ci95_low": float(np.exp(confidence.iloc[0])),
            "ci95_high": float(np.exp(confidence.iloc[1])),
            "p": float(model.summary.loc["score", "p"]),
            "status": "fit",
            "error": "",
        }
    except Exception as exc:  # pragma: no cover - data/version-specific failures.
        result = _empty_cox_result(len(data), events, str(exc))
        result["status"] = "error"
        return result


def _fixed_risk_logrank(
    frame: pd.DataFrame, group_col: str, time_col: str, event_col: str
) -> tuple[float, float]:
    """Return a log-rank statistic for a predeclared risk split."""

    data = frame[[group_col, time_col, event_col]].dropna()
    if data[group_col].nunique() != 2:
        return float("nan"), float("nan")
    group_labels = sorted(data[group_col].unique().tolist())
    try:
        from lifelines.statistics import logrank_test

        left = data[data[group_col] == group_labels[0]]
        right = data[data[group_col] == group_labels[1]]
        result = logrank_test(
            left[time_col],
            right[time_col],
            event_observed_A=left[event_col],
            event_observed_B=right[event_col],
        )
        return float(result.test_statistic), float(result.p_value)
    except ModuleNotFoundError:
        # Keep the endpoint CLI usable without the optional lifelines package.
        # This is the two-group Mantel-Haenszel log-rank statistic with the
        # same event-at-time handling as prognosis.utils.cox_log_rank.
        from scipy.stats import chi2

        times = pd.to_numeric(data[time_col], errors="coerce").to_numpy()
        events = pd.to_numeric(data[event_col], errors="coerce").to_numpy()
        groups = data[group_col].to_numpy() == group_labels[1]
        valid = np.isfinite(times) & np.isfinite(events) & np.isin(events, [0, 1])
        times = times[valid]
        events = events[valid].astype(int)
        groups = groups[valid]
        observed = expected = variance = 0.0
        for event_time in np.unique(times[events == 1]):
            at_risk = times >= event_time
            at_time = (times == event_time) & (events == 1)
            n_total = int(at_risk.sum())
            n_group = int((at_risk & groups).sum())
            event_count = int(at_time.sum())
            group_events = int((at_time & groups).sum())
            if n_total <= 1:
                continue
            observed += group_events
            expected += event_count * n_group / n_total
            variance += (
                n_group
                * (n_total - n_group)
                * event_count
                * (n_total - event_count)
                / (n_total * n_total * (n_total - 1))
            )
        if variance <= 0:
            return float("nan"), float("nan")
        statistic = float((observed - expected) ** 2 / variance)
        return statistic, float(chi2.sf(statistic, 1))
    except Exception:  # pragma: no cover - data/version-specific failures.
        return float("nan"), float("nan")


def fixed_os_risk_pfi_analysis(
    frame: pd.DataFrame,
    *,
    risk_col: str = "hazard_pred",
    pfi_event_col: str = "PFI",
    pfi_time_col: str = "PFI.time",
    patient_col: Optional[str] = None,
    cutoff: Optional[float] = None,
    time_days_per_month: float = 30.4375,
) -> tuple[pd.DataFrame, dict]:
    """Analyze PFI with a fixed OS-trained risk score.

    The median risk cutoff is calculated from the supplied risk table only and
    is never optimized against PFI. The function has no model or optimizer
    argument, making accidental PFI retraining difficult.
    """

    required = [risk_col, pfi_event_col, pfi_time_col]
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise KeyError(f"Missing required PFI columns: {missing}")
    if time_days_per_month <= 0:
        raise ValueError("time_days_per_month must be positive.")

    risk = pd.to_numeric(frame[risk_col], errors="coerce")
    event = _event_values(frame[pfi_event_col], pfi_event_col)
    time_days = pd.to_numeric(frame[pfi_time_col], errors="coerce")
    finite_risk = risk.notna() & np.isfinite(risk)
    valid = finite_risk & event.notna() & time_days.notna()
    valid &= np.isfinite(time_days) & (time_days > 0)
    matched = frame.loc[valid].copy()
    if matched.empty:
        raise ValueError("No valid risk/PFI rows remain after endpoint validation.")

    matched[risk_col] = risk.loc[valid].astype(float)
    matched[pfi_event_col] = event.loc[valid].astype(int)
    matched[pfi_time_col] = time_days.loc[valid].astype(float)
    # Keep the cutoff and z-score anchored to the complete OS risk table.
    # PFI missingness must not change the prespecified OS risk definition.
    risk_cutoff = float(np.median(risk.loc[finite_risk])) if cutoff is None else float(cutoff)
    if not np.isfinite(risk_cutoff):
        raise ValueError("The fixed OS-risk cutoff must be finite.")
    risk_z = _safe_zscore(risk)
    matched["hazard_pred_z"] = risk_z.loc[valid]
    matched["fixed_os_median_cutoff"] = risk_cutoff
    matched["fixed_os_median_group"] = np.where(
        matched[risk_col] > risk_cutoff, "High", "Low"
    )
    matched["pfi_time_days"] = matched[pfi_time_col]
    matched["pfi_time_months"] = matched[pfi_time_col] / time_days_per_month
    if patient_col and patient_col in matched.columns:
        matched["patient_id"] = matched[patient_col].astype(str)
    elif "patient_id" not in matched.columns:
        fallback = next(
            (name for name in ("patient_barcode", "bcr_patient_barcode", "WSIs") if name in matched),
            None,
        )
        if fallback:
            matched["patient_id"] = matched[fallback].astype(str)

    cox = _fit_cox_per_sd(matched, "pfi_time_days", pfi_event_col, "hazard_pred_z")
    chi2, logrank_p = _fixed_risk_logrank(
        matched, "fixed_os_median_group", "pfi_time_days", pfi_event_col
    )
    cindex = cindex_lifeline(
        matched[risk_col].to_numpy(),
        matched[pfi_event_col].to_numpy(),
        matched[pfi_time_col].to_numpy(),
    )
    summary = {
        "risk_n": int(len(frame)),
        "matched_valid_pfi_n": int(len(matched)),
        "unmatched_or_invalid_pfi_n": int(len(frame) - len(matched)),
        "pfi_events": int(matched[pfi_event_col].sum()),
        "pfi_censored": int((matched[pfi_event_col] == 0).sum()),
        "cox_hr_per_sd": cox["hr_per_sd"],
        "cox_ci95_low": cox["ci95_low"],
        "cox_ci95_high": cox["ci95_high"],
        "cox_p": cox["p"],
        "pfi_cindex": cindex,
        "km_cutoff_hazard_pred": risk_cutoff,
        "km_low_n": int((matched["fixed_os_median_group"] == "Low").sum()),
        "km_high_n": int((matched["fixed_os_median_group"] == "High").sum()),
        "km_low_events": int(
            matched.loc[matched["fixed_os_median_group"] == "Low", pfi_event_col].sum()
        ),
        "km_high_events": int(
            matched.loc[matched["fixed_os_median_group"] == "High", pfi_event_col].sum()
        ),
        "logrank_chi2": chi2,
        "logrank_p": logrank_p,
        "cox_status": cox["status"],
        "cox_error": cox["error"],
        "analysis_scope": "fixed OS-trained risk score; no PFI-specific retraining or tuning",
    }
    return matched, summary


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


def _read_table(path: Path, sheet: Optional[str] = None) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Input table does not exist: {path}")
    if path.suffix.lower() == ".xlsx":
        return pd.read_excel(path, sheet_name=sheet or 0)
    return pd.read_csv(path)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, default=_json_default) + "\n", encoding="utf-8"
    )


def _run_three_year(args: argparse.Namespace) -> None:
    frame = _read_table(args.input_csv)
    labels = assign_three_year_os_groups(
        frame,
        time_col=args.time_col,
        event_col=args.event_col,
        threshold_months=args.threshold_months,
    )
    output = frame.copy()
    output[args.output_col] = labels
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_dir / "os_3year_patient_table.csv", index=False)
    counts = output[args.output_col].value_counts(dropna=False).to_dict()
    summary = {
        "threshold_months": args.threshold_months,
        "rule": "event and time <= threshold is early death; time > threshold is survived beyond threshold; censored at or before threshold is excluded",
        "counts": counts,
    }
    _write_json(args.output_dir / "os_3year_summary.json", summary)
    print(json.dumps(summary, indent=2, default=_json_default))


def _run_pfi(args: argparse.Namespace) -> None:
    risk = _read_table(args.risk_csv)
    endpoint = _read_table(args.endpoint_csv, args.endpoint_sheet)
    if args.risk_id_col not in risk.columns or args.endpoint_id_col not in endpoint.columns:
        raise KeyError("Both PFI input tables must contain their declared identifier columns.")
    risk_key = risk[args.risk_id_col].astype(str).str.strip()
    endpoint_key = endpoint[args.endpoint_id_col].astype(str).str.strip()
    if risk_key.duplicated().any() or endpoint_key.duplicated().any():
        raise ValueError("PFI risk and endpoint identifier columns must be unique before merging.")

    missing_endpoint = [
        name
        for name in (args.pfi_event_col, args.pfi_time_col)
        if name not in endpoint.columns
    ]
    if missing_endpoint:
        raise KeyError(f"PFI endpoint table is missing required columns: {missing_endpoint}")

    # Merge only the endpoint fields needed for this analysis. Internal names
    # prevent a stale same-named column in the risk table from being selected.
    endpoint_payload = endpoint[
        [args.endpoint_id_col, args.pfi_event_col, args.pfi_time_col]
    ].copy()
    endpoint_payload = endpoint_payload.rename(
        columns={
            args.endpoint_id_col: "_morphx_endpoint_id",
            args.pfi_event_col: "_morphx_pfi_event",
            args.pfi_time_col: "_morphx_pfi_time",
        }
    )
    merged = risk.assign(_morphx_id=risk_key).merge(
        endpoint_payload.assign(_morphx_id=endpoint_key),
        on="_morphx_id",
        how="left",
    )
    merged[args.pfi_event_col] = merged.pop("_morphx_pfi_event")
    merged[args.pfi_time_col] = merged.pop("_morphx_pfi_time")
    merged = merged.drop(columns=["_morphx_id", "_morphx_endpoint_id"])
    matched, summary = fixed_os_risk_pfi_analysis(
        merged,
        risk_col=args.risk_col,
        pfi_event_col=args.pfi_event_col,
        pfi_time_col=args.pfi_time_col,
        patient_col=args.risk_id_col,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    matched.to_csv(args.output_dir / "pfi_fixed_risk_patient_table.csv", index=False)
    _write_json(args.output_dir / "pfi_fixed_risk_summary.json", summary)
    print(json.dumps(summary, indent=2, default=_json_default))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run fixed secondary endpoint analyses for MorphX."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    os_parser = subparsers.add_parser(
        "three-year-os", help="Assign the prespecified 3-year OS groups."
    )
    os_parser.add_argument("--input-csv", type=Path, required=True)
    os_parser.add_argument("--time-col", default="os_time_months")
    os_parser.add_argument("--event-col", default="os_status")
    os_parser.add_argument("--threshold-months", type=float, default=36.0)
    os_parser.add_argument("--output-col", default="os_3year_subgroup")
    os_parser.add_argument("--output-dir", type=Path, required=True)
    os_parser.set_defaults(handler=_run_three_year)

    pfi_parser = subparsers.add_parser(
        "pfi", help="Analyze PFI with a fixed OS-trained risk score."
    )
    pfi_parser.add_argument("--risk-csv", type=Path, required=True)
    pfi_parser.add_argument("--endpoint-csv", "--endpoint-table", dest="endpoint_csv",
                            type=Path, required=True)
    pfi_parser.add_argument("--endpoint-sheet", default="TCGA-CDR",
                            help="Sheet name when the PFI endpoint input is an Excel workbook.")
    pfi_parser.add_argument("--risk-id-col", default="patient_barcode")
    pfi_parser.add_argument("--endpoint-id-col", default="patient_barcode")
    pfi_parser.add_argument("--risk-col", default="hazard_pred")
    pfi_parser.add_argument("--pfi-event-col", default="PFI")
    pfi_parser.add_argument("--pfi-time-col", default="PFI.time")
    pfi_parser.add_argument("--output-dir", type=Path, required=True)
    pfi_parser.set_defaults(handler=_run_pfi)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
