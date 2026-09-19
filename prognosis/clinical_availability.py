"""Report clinical covariate and treatment-field availability without patient rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from configs.cohorts import default_clinical_path, default_wsi_list_path, get_cohort
from prognosis.clinical import wsi_identifier


FIELD_ALIASES = {
    "age": ("Age", "age"),
    "sex": ("Gender", "sex"),
    "stage": ("ajcc_pathologic_stage", "stage", "overall_stage", "tumor_stage"),
    "tnm": ("ajcc_pathologic_t", "TNMstage"),
    "grade": ("poorly_differentiated", "grade"),
    "afp": ("AFP",),
    "smoking": ("Smoking", "smoke"),
    "vascular_invasion": ("Vessel invasion", "vascular_invasion"),
    "pleural_involvement": ("pleural_involvement",),
    "treatment": ("prior_treatment",),
}


def summarize(cohort: str, dataset: str, data_root: Path,
              clinical_csv: Path | None = None, wsi_list: Path | None = None) -> dict:
    spec = get_cohort(cohort)
    if dataset not in spec.clinical_files:
        raise ValueError(f"{dataset} is not declared for {cohort}")
    list_path = wsi_list or default_wsi_list_path(data_root, spec, dataset)
    with list_path.open(encoding="utf-8-sig") as handle:
        names = json.load(handle)
    if not isinstance(names, list) or not names:
        raise ValueError(f"Expected a nonempty WSI list: {list_path}")
    source = clinical_csv or default_clinical_path(data_root, spec, dataset)
    clinical = pd.read_csv(source)
    identifier = next((name for name in ("WSIs", "wsi_id", "WSI", "case_id", "patient_id",
                                         "submitter_id", "Sample ID") if name in clinical), None)
    if identifier is None:
        raise ValueError(f"No clinical identifier column in {source}")
    clinical["_id"] = clinical[identifier].astype(str).str.strip()
    clinical = clinical.drop_duplicates("_id", keep="first").set_index("_id")
    ids = [wsi_identifier(name, spec, dataset) for name in names]
    missing = [value for value in ids if value not in clinical.index]
    if missing:
        raise ValueError(f"{len(missing)} WSI identifiers have no clinical match")
    matched = clinical.loc[ids]
    fields = {}
    for field, aliases in FIELD_ALIASES.items():
        present = [name for name in aliases if name in matched.columns]
        fields[field] = {name: {"available": int(matched[name].notna().sum()),
                                "missing": int(matched[name].isna().sum())}
                         for name in present}
        if not present:
            fields[field] = {"status": "column_absent"}
    return {"cohort": cohort, "dataset": dataset, "n_wsi": len(names),
            "n_unique_clinical_ids": len(set(ids)), "fields": fields,
            "treatment_detail_available": bool(FIELD_ALIASES["treatment"][0] in matched.columns and
                                               matched[FIELD_ALIASES["treatment"][0]].notna().any())}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Summarize clinical and treatment field availability.")
    parser.add_argument("--cohort", choices=("hcc", "luad"), required=True)
    parser.add_argument("--dataset", choices=("tcga", "kmmufh", "gd"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--clinical-csv", type=Path)
    parser.add_argument("--wsi-list", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args(argv)
    report = summarize(args.cohort, args.dataset, args.data_root, args.clinical_csv, args.wsi_list)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
