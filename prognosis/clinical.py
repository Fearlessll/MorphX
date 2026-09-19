"""Clinical endpoint loading for HCC and LUAD.

All returned event indicators use the same convention: ``1`` means death or
the observed event, and ``0`` means right censoring.
"""

from pathlib import Path
from typing import Iterable, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd

from configs.cohorts import CohortSpec, default_clinical_path, get_cohort


def _strip_name(value: object) -> str:
    return Path(str(value)).name


def _wsi_id(wsi_name: object, cohort: CohortSpec, dataset: str) -> str:
    """Map a feature-map filename to the identifier used in clinical tables."""

    name = Path(_strip_name(wsi_name)).stem
    dataset = dataset.lower()
    if cohort.name == "hcc":
        if dataset == "tcga" and name.startswith("TCGA"):
            return name[:23]
        return name.split("-", 1)[0]
    if dataset == "tcga":
        return name[:12]
    return name.split("-", 1)[0]


def _find_column(frame: pd.DataFrame, candidates: Iterable[str], label: str) -> str:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise ValueError(
        f"Clinical CSV is missing the {label} column. "
        f"Expected one of {list(candidates)}; found {list(frame.columns)}"
    )


def _as_numeric(series: pd.Series, name: str) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce")
    if values.isna().any():
        bad = series[values.isna()].head(5).tolist()
        raise ValueError(f"Column {name!r} contains non-numeric values, examples: {bad}")
    return values.to_numpy(dtype=np.float64)


def _as_event(series: pd.Series, name: str) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce")
    if values.isna().any():
        normalized = series.astype(str).str.strip().str.lower()
        mapping = {
            "1": 1,
            "0": 0,
            "true": 1,
            "false": 0,
            "yes": 1,
            "no": 0,
            "dead": 1,
            "alive": 0,
            "deceased": 1,
            "living": 0,
        }
        values = normalized.map(mapping)
    if values.isna().any():
        bad = series[values.isna()].head(5).tolist()
        raise ValueError(f"Column {name!r} contains unsupported event values: {bad}")
    values = values.astype(float)
    if not values.isin([0.0, 1.0]).all():
        raise ValueError(f"Column {name!r} must contain binary event values.")
    return values.to_numpy(dtype=np.int64)


def _as_vital_status(series: pd.Series, name: str) -> np.ndarray:
    """Normalize the historical HCC TCGA vital-status convention.

    The original HCC tables use ``1`` for alive/censored and ``0`` for dead.
    Text exports may instead contain ``Alive`` and ``Dead``.
    """

    values = pd.to_numeric(series, errors="coerce")
    if values.isna().any():
        normalized = series.astype(str).str.strip().str.lower()
        values = normalized.map(
            {
                "alive": 1,
                "living": 1,
                "censored": 1,
                "dead": 0,
                "deceased": 0,
            }
        )
    if values.isna().any():
        bad = series[values.isna()].head(5).tolist()
        raise ValueError(f"Column {name!r} contains unsupported vital-status values: {bad}")
    values = values.astype(float)
    if not values.isin([0.0, 1.0]).all():
        raise ValueError(f"Column {name!r} must contain binary vital-status values.")
    return values.to_numpy(dtype=np.int64)


def _select_survival_days(
    event: np.ndarray,
    event_days: pd.Series,
    censor_days: pd.Series,
) -> np.ndarray:
    """Select only the relevant time column for each sample.

    TCGA clinical exports commonly leave the non-applicable time column empty.
    """

    event_values = pd.to_numeric(event_days, errors="coerce").to_numpy(dtype=np.float64)
    censor_values = pd.to_numeric(censor_days, errors="coerce").to_numpy(dtype=np.float64)
    selected = np.where(event.astype(bool), event_values, censor_values)
    if not np.isfinite(selected).all():
        bad = np.where(~np.isfinite(selected))[0].tolist()
        raise ValueError(f"Selected survival times contain missing or non-numeric values: {bad[:5]}")
    return selected


def load_survival_labels(
    wsi_names: Iterable[str],
    cohort: Union[str, CohortSpec],
    dataset: str,
    data_root: Optional[Union[str, Path]] = None,
    clinical_csv: Optional[Union[str, Path]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load event indicators and survival times in months.

    ``clinical_csv`` is preferred for reproducibility. ``data_root`` is only
    used to construct the documented default ``<data_root>/<dataset>`` path.
    """

    spec = get_cohort(cohort) if isinstance(cohort, str) else cohort
    dataset_key = str(dataset).lower()
    if clinical_csv is None:
        if data_root is None:
            raise ValueError("Provide either clinical_csv or data_root.")
        clinical_csv = default_clinical_path(Path(data_root), spec, dataset_key)
    clinical_csv = Path(clinical_csv)
    if not clinical_csv.is_file():
        raise FileNotFoundError(f"Clinical CSV does not exist: {clinical_csv}")

    frame = pd.read_csv(clinical_csv)
    id_column = _find_column(
        frame,
        ("WSIs", "wsi_id", "WSI", "case_id", "patient_id", "submitter_id", "Sample ID"),
        "patient/WSI identifier",
    )
    ids = frame[id_column].astype(str).str.strip()
    requested_ids = [_wsi_id(name, spec, dataset_key) for name in wsi_names]
    duplicate_ids = set(ids[ids.duplicated(keep=False)]) & set(requested_ids)
    if duplicate_ids:
        if spec.name != "hcc" or dataset_key != "kmmufh":
            raise ValueError(f"Clinical CSV has duplicate requested identifiers: {sorted(duplicate_ids)[:5]}")
        warnings.warn(
            f"KMMUFH clinical CSV contains duplicate requested IDs {sorted(duplicate_ids)[:5]}; "
            "using the first row as in the source implementation. Review conflicting records.",
            RuntimeWarning,
            stacklevel=2,
        )
    clinical = frame.copy()
    clinical["_morphx_id"] = ids
    clinical = clinical.drop_duplicates("_morphx_id", keep="first")
    clinical = clinical.set_index("_morphx_id", drop=False)

    missing = [item for item in requested_ids if item not in clinical.index]
    if missing:
        raise KeyError(
            f"{len(missing)} feature maps have no clinical match in {clinical_csv}. "
            f"Examples: {missing[:5]}"
        )
    selected = clinical.loc[requested_ids]

    if spec.name == "hcc" and dataset_key == "tcga":
        vital = _as_vital_status(
            selected[_find_column(selected, ("vital_status",), "HCC TCGA vital status")],
            "vital_status",
        )
        event = 1 - vital  # Historical HCC TCGA files encode 1 as alive/censored.
        days = _select_survival_days(
            event,
            selected["days_to_death"],
            selected["days_to_last_follow_up"],
        )
    elif spec.name == "luad" and dataset_key == "tcga":
        event = _as_event(
            selected[_find_column(selected, ("OS_status",), "LUAD TCGA OS status")],
            "OS_status",
        )
        days = _select_survival_days(
            event,
            selected["days_to_death"],
            selected["days_to_follow_up"],
        )
    elif spec.name == "luad" and dataset_key == "gd":
        event = _as_event(
            selected[_find_column(selected, ("OS_status",), "LUAD GD OS status")],
            "OS_status",
        )
        days = _as_numeric(selected["OS_month"], "OS_month") * 30.0
    elif spec.name == "hcc" and dataset_key == "kmmufh":
        event = _as_event(
            selected[_find_column(selected, ("OS_status",), "HCC KMMUFH OS status")],
            "OS_status",
        )
        days = _as_numeric(selected["OS"], "OS")
    else:
        raise ValueError(f"Unsupported cohort/dataset pair: {spec.name}/{dataset_key}")

    time_months = np.asarray(days, dtype=np.float64) / 30.0
    if not np.isfinite(time_months).all() or (time_months < 0).any():
        bad = np.where(~np.isfinite(time_months) | (time_months < 0))[0].tolist()
        raise ValueError(f"Survival times must be nonnegative and finite; invalid rows: {bad[:5]}")
    return event.astype(np.float32), time_months.astype(np.float32)


def wsi_identifier(
    wsi_name: object, cohort: Union[str, CohortSpec], dataset: str
) -> str:
    """Return the clinical identifier for one feature-map filename."""

    spec = get_cohort(cohort) if isinstance(cohort, str) else cohort
    return _wsi_id(wsi_name, spec, dataset)
