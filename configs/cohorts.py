"""Cohort-specific conventions used by MorphX.

The feature-map channel order is part of the data contract.  Do not infer it
from filenames: HCC and LUAD use different tissue-channel counts.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class CohortSpec:
    """Static conventions for one cancer cohort."""

    name: str
    tissue_channels: int
    texture_channels: int
    nuclear_channels: int
    tissue_labels: Tuple[str, ...]
    clinical_files: Dict[str, str]
    wsi_list_files: Dict[str, str]
    feature_name_file: str
    backbone_width: int
    macro_epochs: int
    fusion_epochs: int
    macro_learning_rate: float
    fusion_learning_rate: float
    discovery_dataset: str = "tcga"
    external_dataset: str = ""

    @property
    def total_channels(self) -> int:
        return self.tissue_channels + self.texture_channels + self.nuclear_channels

    @property
    def family_sizes(self) -> Dict[str, int]:
        return {
            "tissue": self.tissue_channels,
            "texture": self.texture_channels,
            "nuclear": self.nuclear_channels,
        }

    @property
    def family_slices(self) -> Dict[str, slice]:
        tissue_end = self.tissue_channels
        texture_end = tissue_end + self.texture_channels
        return {
            "tissue": slice(0, tissue_end),
            "texture": slice(tissue_end, texture_end),
            "nuclear": slice(texture_end, self.total_channels),
        }

    def validate_training_dataset(self, dataset: str) -> None:
        """Keep model fitting on the declared discovery cohort."""

        if str(dataset).strip().lower() != self.discovery_dataset:
            raise ValueError(
                f"Training is restricted to {self.discovery_dataset.upper()} for "
                f"{self.name.upper()}; use external evaluation for "
                f"{self.external_dataset.upper()}."
            )

    def validate_external_dataset(self, dataset: str) -> None:
        """Require the predeclared external-validation cohort."""

        if str(dataset).strip().lower() != self.external_dataset:
            raise ValueError(
                f"Expected locked external dataset {self.external_dataset!r} for "
                f"{self.name.upper()}, got {dataset!r}."
            )


HCC = CohortSpec(
    name="hcc",
    tissue_channels=8,
    texture_channels=40,
    nuclear_channels=120,
    tissue_labels=(
        "tumor",
        "empty",
        "fibrosis",
        "inflammation",
        "necrosis",
        "normal",
        "reaction",
        "steatosis",
    ),
    clinical_files={
        "tcga": "TCGA.csv",
        "kmmufh": "KMMUFH_data_label_fix.csv",
    },
    wsi_list_files={"tcga": "all_data.json", "kmmufh": "all_data.json"},
    feature_name_file="feats_HCC.cvs",
    backbone_width=128,
    macro_epochs=15,
    fusion_epochs=10,
    macro_learning_rate=6.6e-4,
    fusion_learning_rate=6.6e-5,
    external_dataset="kmmufh",
)

LUAD = CohortSpec(
    name="luad",
    tissue_channels=5,
    texture_channels=40,
    nuclear_channels=120,
    tissue_labels=("nec", "empty", "te", "tas", "lym"),
    clinical_files={
        "tcga": "TCGA_clinical_data_new.csv",
        "gd": "GD_clinical_data.csv",
    },
    wsi_list_files={"tcga": "all_data.json", "gd": "all_data_mil.json"},
    feature_name_file="feats_LUNG.cvs",
    backbone_width=256,
    macro_epochs=40,
    fusion_epochs=20,
    macro_learning_rate=6.6e-4,
    fusion_learning_rate=9.9e-5,
    external_dataset="gd",
)

_COHORTS = {"hcc": HCC, "luad": LUAD}


def get_cohort(name: str) -> CohortSpec:
    """Return a cohort specification using a case-insensitive name."""

    key = str(name).strip().lower()
    if key not in _COHORTS:
        raise ValueError(f"Unknown cohort {name!r}; expected one of {sorted(_COHORTS)}")
    return _COHORTS[key]


def parse_feature_families(value: str, cohort: CohortSpec) -> Tuple[List[int], List[str]]:
    """Resolve ``all`` or a legacy ``8+40+120`` style selection.

    Returns the selected absolute channel indices and the selected family names
    in canonical tissue/texture/nuclear order.
    """

    raw = str(value).strip().lower()
    if raw in {"all", "full", "all_features"}:
        names = ["tissue", "texture", "nuclear"]
        return list(range(cohort.total_channels)), names

    requested_sizes = []
    for token in raw.split("+"):
        try:
            requested_sizes.append(int(token))
        except ValueError as exc:
            raise ValueError(
                f"Invalid feature-family selection {value!r}; use 'all' or sizes joined by '+'."
            ) from exc

    selected_names: List[str] = []
    indices: List[int] = []
    for family_name in ("tissue", "texture", "nuclear"):
        size = cohort.family_sizes[family_name]
        if size in requested_sizes:
            selected_names.append(family_name)
            family_slice = cohort.family_slices[family_name]
            indices.extend(range(family_slice.start, family_slice.stop))

    expected = sorted(cohort.family_sizes[name] for name in selected_names)
    if sorted(requested_sizes) != expected:
        valid = "+".join(str(cohort.family_sizes[name]) for name in ("tissue", "texture", "nuclear"))
        raise ValueError(
            f"Feature selection {value!r} does not match {cohort.name.upper()} families. "
            f"Valid family combinations use {valid}."
        )
    return indices, selected_names


def default_dataset_dir(data_root: Path, cohort: CohortSpec, dataset: str) -> Path:
    """Return ``data_root/<cohort-specific dataset>`` without hard-coded paths."""

    del cohort
    directory_names = {"tcga": "TCGA", "kmmufh": "KMMUFH", "gd": "GD"}
    dataset_key = str(dataset).lower()
    try:
        directory_name = directory_names[dataset_key]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset {dataset!r}") from exc
    return Path(data_root) / directory_name


def default_clinical_path(data_root: Path, cohort: CohortSpec, dataset: str) -> Path:
    """Return the default clinical CSV path for a dataset."""

    dataset_key = str(dataset).lower()
    try:
        filename = cohort.clinical_files[dataset_key]
    except KeyError as exc:
        raise ValueError(
            f"Dataset {dataset!r} is not defined for cohort {cohort.name!r}; "
            f"available datasets: {sorted(cohort.clinical_files)}"
        ) from exc
    return default_dataset_dir(data_root, cohort, dataset_key) / filename


def default_wsi_list_path(data_root: Path, cohort: CohortSpec, dataset: str) -> Path:
    """Return the paper analysis list, which can differ from all_data.json."""

    dataset_key = str(dataset).lower()
    return default_dataset_dir(data_root, cohort, dataset_key) / cohort.wsi_list_files[dataset_key]
