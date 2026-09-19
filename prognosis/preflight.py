"""Read-only audit of real MorphX cohort inputs before any training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from configs.cohorts import default_dataset_dir, default_wsi_list_path, get_cohort
from prognosis.clinical import load_survival_labels
from prognosis.train_mfm_msfm import _default_feature_map_dir, _load_split, _resolve_feature_paths


def _names(path: Path) -> list[str]:
    with path.open(encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        payload = next((payload[key] for key in ("all_data", "test_data", "data", "wsi_names")
                        if key in payload), payload)
    if not isinstance(payload, list) or not payload or len(set(payload)) != len(payload):
        raise ValueError(f"WSI list is empty, duplicated, or malformed: {path}")
    return [str(item) for item in payload]


def audit(cohort: str, dataset: str, data_root: Path, feature_map_dir: Path | None = None,
          clinical_csv: Path | None = None, wsi_list: Path | None = None,
          splits_dir: Path | None = None, image_size: int = 256) -> dict:
    spec = get_cohort(cohort)
    if dataset not in spec.clinical_files:
        raise ValueError(f"{dataset} is not a declared dataset for {cohort}")
    dataset_dir = default_dataset_dir(data_root, spec, dataset)
    names = _names(wsi_list or default_wsi_list_path(data_root, spec, dataset))
    maps_dir = feature_map_dir or _default_feature_map_dir(data_root, spec, dataset, image_size)
    paths = _resolve_feature_paths(names, maps_dir)
    shapes = {}
    for path in paths:
        values = np.load(path, mmap_mode="r")
        shape = tuple(values.shape)
        shapes[str(shape)] = shapes.get(str(shape), 0) + 1
        if shape != (image_size, image_size, spec.total_channels):
            raise ValueError(f"{path} has {shape}; expected {(image_size, image_size, spec.total_channels)}")
    events, times = load_survival_labels(names, spec, dataset, data_root=data_root,
                                         clinical_csv=clinical_csv)
    report = {"cohort": cohort, "dataset": dataset, "n": len(names),
              "events": int(events.sum()), "zero_month_followup": int((times == 0).sum()),
              "feature_map_shape_counts": shapes, "clinical_match": len(events),
              "feature_map_dir": str(maps_dir), "split_folds": []}
    if dataset == spec.discovery_dataset:
        split_root = splits_dir or dataset_dir
        full_set = set(names)
        validation_counts = dict.fromkeys(names, 0)
        for fold in range(10):
            train, validation = _load_split(split_root, fold)
            if set(train) | set(validation) != full_set:
                raise ValueError(f"Fold {fold} does not partition the discovery WSI list")
            for name in validation:
                validation_counts[name] += 1
            report["split_folds"].append({"fold": fold, "train": len(train),
                                          "validation": len(validation)})
        report["validation_multiplicity"] = {
            str(count): sum(value == count for value in validation_counts.values())
            for count in sorted(set(validation_counts.values()))
        }
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only cohort input and fixed-split audit.")
    parser.add_argument("--cohort", choices=("hcc", "luad"), required=True)
    parser.add_argument("--dataset", choices=("tcga", "kmmufh", "gd"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--feature-map-dir", type=Path)
    parser.add_argument("--clinical-csv", type=Path)
    parser.add_argument("--wsi-list", type=Path)
    parser.add_argument("--splits-dir", type=Path)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--output-json", type=Path)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    report = audit(args.cohort, args.dataset, args.data_root, args.feature_map_dir,
                   args.clinical_csv, args.wsi_list, args.splits_dir, args.image_size)
    encoded = json.dumps(report, indent=2)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
