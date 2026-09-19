"""Load the frozen source channel order from packaged metadata."""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

from .cohorts import get_cohort


@lru_cache(maxsize=2)
def feature_names(cohort: str) -> tuple[str, ...]:
    spec = get_cohort(cohort)
    metadata_path = Path(__file__).resolve().parent / "metadata" / spec.feature_name_file
    with metadata_path.open(encoding="utf-8-sig") as handle:
        names = tuple(next(csv.reader(handle)))
    if len(names) != spec.total_channels:
        raise ValueError(f"{spec.feature_name_file} contains {len(names)} names; expected {spec.total_channels}")
    return names
