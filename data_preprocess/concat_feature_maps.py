"""Concatenate tissue, texture, and nuclear feature-map families.

The output channel order is fixed by :mod:`configs.cohorts`:
``tissue -> texture -> nuclear``.  This order matches the supplied
``feats_HCC.cvs`` and ``feats_LUNG.cvs`` metadata files.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import cv2
import numpy as np

from configs.cohorts import get_cohort, parse_feature_families


def _load_names(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        for key in ("all_data", "data", "wsi_names"):
            if key in payload:
                payload = payload[key]
                break
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list of feature-map filenames.")
    return [str(value) for value in payload]


def _load_family(directory: Path, name: str, expected_channels: int) -> np.ndarray:
    path = directory / name
    if not path.is_file():
        raise FileNotFoundError(f"Missing feature map: {path}")
    values = np.asarray(np.load(path), dtype=np.float32)
    if values.ndim == 2:
        values = values[..., np.newaxis]
    if values.ndim != 3:
        raise ValueError(f"{path} must be HWC, got shape {values.shape}")
    if values.shape[-1] != expected_channels:
        raise ValueError(
            f"{path} has {values.shape[-1]} channels; expected {expected_channels}."
        )
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def _resize(values: np.ndarray, height: int, width: int, interpolation: int) -> np.ndarray:
    if values.shape[:2] == (height, width):
        return values
    return cv2.resize(values, (width, height), interpolation=interpolation)


def concat_feature_maps(
    wsi_names: Iterable[str],
    cohort: str,
    output_dir: Path,
    feature_families: str = "all",
    tissue_dir: Optional[Path] = None,
    texture_dir: Optional[Path] = None,
    nuclear_dir: Optional[Path] = None,
    overwrite: bool = False,
) -> None:
    """Create one ordered HWC feature map per WSI."""

    spec = get_cohort(cohort)
    _, selected_families = parse_feature_families(feature_families, spec)
    directories: Dict[str, Optional[Path]] = {
        "tissue": tissue_dir,
        "texture": texture_dir,
        "nuclear": nuclear_dir,
    }
    for family in selected_families:
        if directories[family] is None:
            raise ValueError(f"--{family}-dir is required for selected family {family!r}.")
    output_dir.mkdir(parents=True, exist_ok=True)

    for name in wsi_names:
        output_path = output_dir / name
        if output_path.exists() and not overwrite:
            continue
        arrays: Dict[str, np.ndarray] = {}
        for family in selected_families:
            arrays[family] = _load_family(
                directories[family], name, spec.family_sizes[family]
            )

        reference = arrays[selected_families[0]]
        height, width = reference.shape[:2]
        channels = []
        for family in selected_families:
            interpolation = cv2.INTER_NEAREST if family == "tissue" else cv2.INTER_LINEAR
            values = _resize(arrays[family], height, width, interpolation)
            channels.append(values)
        merged = np.concatenate(channels, axis=-1).astype(np.float32, copy=False)
        expected = sum(spec.family_sizes[family] for family in selected_families)
        if merged.shape[-1] != expected:
            raise RuntimeError(f"Internal channel mismatch for {name}: {merged.shape[-1]} != {expected}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, merged)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Concatenate MorphX feature-map families.")
    parser.add_argument("--cohort", choices=("hcc", "luad"), required=True)
    parser.add_argument("--wsi-list", type=Path, required=True, help="JSON list such as all_data.json.")
    parser.add_argument("--tissue-dir", type=Path, default=None)
    parser.add_argument("--texture-dir", type=Path, default=None)
    parser.add_argument("--nuclear-dir", type=Path, default=None)
    parser.add_argument("--feature-families", default="all", help="all or legacy sizes, e.g. 8+40.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    concat_feature_maps(
        _load_names(args.wsi_list),
        args.cohort,
        args.output_dir,
        args.feature_families,
        args.tissue_dir,
        args.texture_dir,
        args.nuclear_dir,
        args.overwrite,
    )


if __name__ == "__main__":
    main()
