"""Generate MFM integrated-gradient key tiles with traceable source coordinates.

The geometry JSON is written by ``normalize_feature_maps``. An attribution
cannot be mapped back to a WSI without that file and the feature-map grid
spacing used during extraction.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from configs.cohorts import get_cohort, parse_feature_families
from configs.features import feature_names
from prognosis.data_loaders import _feature_map_tensor
from prognosis.Networks.resnet import resnet10


def integrated_gradients(model, input_tensor: torch.Tensor, steps: int = 50) -> torch.Tensor:
    """Right Riemann IG, matching Captum's default approximation rule."""

    if steps < 1:
        raise ValueError("steps must be positive")
    baseline = torch.zeros_like(input_tensor)
    accumulated = torch.zeros_like(input_tensor)
    model.eval()
    for step in range(1, steps + 1):
        scaled = (baseline + input_tensor * (step / steps)).detach().requires_grad_(True)
        score = model(scaled)[1].sum()
        gradient = torch.autograd.grad(score, scaled)[0]
        accumulated += gradient.detach()
    return (input_tensor - baseline) * accumulated / steps


def model_to_source(row: int, col: int, geometry: dict) -> tuple[int, int] | None:
    """Invert the recorded crop, pad, and resize for one model-map pixel."""

    side = int(geometry["square_side"])
    size = int(geometry["image_size"])
    top, bottom, left, right = map(int, geometry["crop_box"])
    offset_y, offset_x = map(int, geometry["square_offset"])
    source_y = math.floor((row + 0.5) * side / size - offset_y + top)
    source_x = math.floor((col + 0.5) * side / size - offset_x + left)
    if not (top <= source_y < bottom and left <= source_x < right):
        return None
    return source_y, source_x


def select_tiles(contribution: np.ndarray, geometry: dict, top_k: int, separation: int = 2) -> list[dict]:
    """Choose high-attribution locations with source-grid nonmaximum suppression."""

    if top_k < 1:
        raise ValueError("top_k must be positive")
    if contribution.shape != (geometry["image_size"], geometry["image_size"]):
        raise ValueError("Contribution map and geometry image_size disagree")
    selected: list[dict] = []
    for flat in np.argsort(contribution.reshape(-1))[::-1]:
        row, col = np.unravel_index(flat, contribution.shape)
        if not np.isfinite(contribution[row, col]):
            continue
        source = model_to_source(int(row), int(col), geometry)
        if source is None:
            continue
        source_y, source_x = source
        if any(abs(source_y - item["source_row"]) <= separation and
               abs(source_x - item["source_col"]) <= separation for item in selected):
            continue
        selected.append({"rank": len(selected) + 1, "model_row": int(row),
                         "model_col": int(col), "source_row": source_y,
                         "source_col": source_x, "attribution": float(contribution[row, col])})
        if len(selected) == top_k:
            break
    if len(selected) != top_k:
        raise ValueError(f"Only {len(selected)} valid separated locations; expected {top_k}")
    return selected


def feature_scores(attribution: torch.Tensor, input_tensor: torch.Tensor,
                   tissue_channels: int, names: list[str], wsi_id: str) -> list[dict]:
    """Summarize absolute IG and feature values globally and by tissue class."""

    attr = attribution.squeeze(0).abs().detach().cpu().numpy()
    values = input_tensor.squeeze(0).detach().cpu().numpy()
    if len(names) != attr.shape[0] or attr.shape[0] <= tissue_channels:
        raise ValueError("Full feature map and channel-name metadata are required for biomarker scores")
    tissue_values = values[:tissue_channels]
    assigned = np.argmax(tissue_values, axis=0)
    empty_index = names[:tissue_channels].index("empty")
    valid = (tissue_values.sum(axis=0) > 0.5) & (assigned != empty_index)
    scopes = [("all", valid)] + [(names[i], valid & (assigned == i))
                                 for i in range(tissue_channels) if i != empty_index]
    output = []
    for scope, mask in scopes:
        present = int(mask.sum())
        for channel in range(tissue_channels, attr.shape[0]):
            output.append({"wsi_id": wsi_id, "scope": scope, "feature_index": channel,
                           "feature_name": names[channel], "n_pixels": present,
                           "attribution_sum": float(attr[channel, mask].sum()) if present else 0.0,
                           "feature_mean": float(values[channel, mask].mean()) if present else np.nan})
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Attribute fixed MFM and select WSI key-tile centers.")
    parser.add_argument("--cohort", choices=("hcc", "luad"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--wsi-list", type=Path, required=True)
    parser.add_argument("--feature-map-dir", type=Path, required=True)
    parser.add_argument("--geometry-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--feature-families", default="all")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--feature-scores-csv", type=Path,
                        help="Optional per-slide, per-tissue biomarker attribution and feature values.")
    parser.add_argument("--feature-names", type=Path,
                        help="Feature-name CSV; defaults to the cohort metadata in this repository.")
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    spec = get_cohort(args.cohort)
    channels, _ = parse_feature_families(args.feature_families, spec)
    with args.wsi_list.open(encoding="utf-8-sig") as handle:
        names = json.load(handle)
    if isinstance(names, dict):
        names = next((names[key] for key in ("all_data", "data", "wsi_names") if key in names), names)
    if not isinstance(names, list) or not names:
        raise ValueError("--wsi-list must contain a nonempty JSON list")
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    model = resnet10(first_covd_param=[3, 2, 1], input_channel_num=len(channels),
                     backbone_width=spec.backbone_width, output_use_sigmoid=True).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if "config" in checkpoint:
        saved = checkpoint["config"]
        for key, actual in (("cohort", args.cohort), ("model", "macro"),
                            ("feature_families", args.feature_families)):
            if saved.get(key) != actual:
                raise ValueError(f"Checkpoint {key}={saved.get(key)!r}, requested {actual!r}")
        model.output_use_sigmoid = saved.get("train_output", "sigmoid") == "sigmoid"
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    records = []
    biomarker_records = []
    if args.feature_scores_csv:
        if len(channels) != spec.total_channels:
            raise ValueError("Feature-level analysis requires --feature-families all")
        if args.feature_names:
            import csv

            with args.feature_names.open(encoding="utf-8-sig") as handle:
                selected_feature_names = next(csv.reader(handle))
        else:
            selected_feature_names = feature_names(args.cohort)
        if len(selected_feature_names) != spec.total_channels:
            raise ValueError(f"Feature metadata has {len(selected_feature_names)} columns, "
                             f"expected {spec.total_channels}")
    for name in names:
        map_path = args.feature_map_dir / Path(name).name
        geometry_path = args.geometry_dir / f"{Path(name).stem}.json"
        geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
        tensor = _feature_map_tensor(map_path, channels, geometry["image_size"]).unsqueeze(0).to(device)
        attr = integrated_gradients(model, tensor, args.steps)
        contribution = attr.abs().sum(dim=1).squeeze(0).detach().cpu().numpy()
        tissue_values = tensor[0, :spec.tissue_channels].detach().cpu().numpy()
        empty_index = spec.tissue_labels.index("empty")
        assigned = np.argmax(tissue_values, axis=0)
        contribution[(tissue_values.sum(axis=0) <= 0.5) | (assigned == empty_index)] = -np.inf
        for row in select_tiles(contribution, geometry, args.top_k):
            records.append({"wsi_id": Path(name).name, **row})
        if args.feature_scores_csv:
            biomarker_records.extend(feature_scores(attr, tensor, spec.tissue_channels,
                                                     selected_feature_names, Path(name).name))
        print(f"Selected {args.top_k} key tiles: {name}")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(args.output_csv, index=False)
    if args.feature_scores_csv:
        args.feature_scores_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(biomarker_records).to_csv(args.feature_scores_csv, index=False)


if __name__ == "__main__":
    main()
