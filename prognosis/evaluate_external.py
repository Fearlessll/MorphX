"""Evaluate a fixed MorphX checkpoint on a locked external cohort."""

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from configs.cohorts import default_dataset_dir, default_wsi_list_path, get_cohort, parse_feature_families
from prognosis.clinical import load_survival_labels
from prognosis.data_loaders import FusionFeatureMapDataset, MacroFeatureMapDataset
from prognosis.train_mfm_msfm import (
    _default_feature_map_dir,
    _json_default,
    _resolve_feature_paths,
    _seed_everything,
    _build_model,
)
from prognosis.utils import cindex_lifeline, cox_log_rank


def _load_names(path: Path):
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        payload = next(
            (payload[key] for key in ("all_data", "test_data", "data", "wsi_names") if key in payload),
            payload,
        )
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list of feature-map names.")
    return [str(value) for value in payload]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a fixed MorphX external-validation checkpoint.")
    parser.add_argument("--cohort", choices=("hcc", "luad"), required=True)
    parser.add_argument("--dataset", choices=("kmmufh", "gd"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--wsi-list", type=Path, default=None)
    parser.add_argument("--clinical-csv", type=Path, default=None)
    parser.add_argument("--feature-map-dir", type=Path, default=None)
    parser.add_argument("--feature-families", default="all")
    parser.add_argument("--model", choices=("macro", "fusion"), default="macro")
    parser.add_argument("--micro-features-csv", type=Path, default=None)
    parser.add_argument("--micro-feature-dim", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--first-conv", nargs=3, type=int, default=(3, 2, 1))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    spec = get_cohort(args.cohort)
    spec.validate_external_dataset(args.dataset)
    if args.model == "fusion" and (args.micro_features_csv is None or args.micro_feature_dim is None):
        raise ValueError("Fusion evaluation requires --micro-features-csv and --micro-feature-dim.")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {args.checkpoint}")

    _seed_everything(2024)
    channel_indices, family_names = parse_feature_families(args.feature_families, spec)
    dataset_dir = default_dataset_dir(args.data_root, spec, args.dataset)
    wsi_list = args.wsi_list or default_wsi_list_path(args.data_root, spec, args.dataset)
    names = _load_names(wsi_list)
    feature_map_dir = args.feature_map_dir or _default_feature_map_dir(
        args.data_root, spec, args.dataset, args.image_size
    )
    paths = _resolve_feature_paths(names, feature_map_dir)
    events, times = load_survival_labels(
        names, spec, args.dataset, data_root=args.data_root, clinical_csv=args.clinical_csv
    )
    if args.model == "macro":
        dataset = MacroFeatureMapDataset(
            paths, events, times, channel_indices=channel_indices, image_size=args.image_size
        )
    else:
        dataset = FusionFeatureMapDataset(
            paths,
            events,
            times,
            args.micro_features_csv,
            args.micro_feature_dim,
            channel_indices=channel_indices,
            image_size=args.image_size,
        )

    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    model = _build_model(args, len(channel_indices)).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if isinstance(checkpoint, dict) and "config" in checkpoint:
        saved = checkpoint["config"]
        for key, actual in (("cohort", args.cohort), ("model", args.model),
                            ("feature_families", args.feature_families),
                            ("micro_feature_dim", args.micro_feature_dim)):
            if saved.get(key) != actual:
                raise ValueError(f"Checkpoint {key}={saved.get(key)!r}, requested {actual!r}.")
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state)
    model.output_use_sigmoid = False
    model.eval()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    risks = []
    with torch.no_grad():
        for inputs, _, _ in loader:
            if args.model == "macro":
                scores = model(inputs.to(device).float())[1]
            else:
                micro, macro = inputs
                scores = model(micro.to(device).float(), macro.to(device).float())
            risks.append(scores.reshape(-1).cpu().numpy())
    risks = np.concatenate(risks)
    metrics = {
        "cohort": args.cohort,
        "dataset": args.dataset,
        "model": args.model,
        "feature_families": family_names,
        "channels": len(channel_indices),
        "n": len(names),
        "events": int(events.sum()),
        "cindex": cindex_lifeline(risks, events, times),
        "logrank_p": cox_log_rank(risks, events, times),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "wsi_name": [Path(name).name for name in names],
            "risk_score": risks,
            "survival_months": times,
            "event": events.astype(int),
        }
    ).to_csv(args.output_dir / "external_predictions.csv", index=False)
    with (args.output_dir / "external_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, default=_json_default)
    print(json.dumps(metrics, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
