"""Normalize, crop, pad, and resize feature maps without changing anatomy."""

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from configs.cohorts import get_cohort


def _load_names(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8-sig") as handle:
        values = json.load(handle)
    if isinstance(values, dict):
        values = next(
            (values[key] for key in ("all_data", "data", "wsi_names") if key in values),
            values,
        )
    if not isinstance(values, list):
        raise ValueError(f"{path} must contain a JSON list.")
    return [str(value) for value in values]


def _mask_path(mask_dir: Path, template: str, name: str) -> Path:
    file_name = Path(name).name
    stem = Path(file_name).stem
    return mask_dir / template.format(name=file_name, stem=stem)


def _load_map_and_mask(input_dir: Path, mask_dir: Path, template: str, name: str):
    map_path = input_dir / name
    if not map_path.is_file():
        raise FileNotFoundError(f"Feature map does not exist: {map_path}")
    mask_path = _mask_path(mask_dir, template, name)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Tissue mask does not exist or cannot be read: {mask_path}")
    values = np.asarray(np.load(map_path), dtype=np.float32)
    if values.ndim != 3:
        raise ValueError(f"{map_path} must be HWC, got {values.shape}")
    mask = cv2.resize(mask, (values.shape[1], values.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask = mask > 0
    if not mask.any():
        raise ValueError(f"Tissue mask has no positive pixels: {mask_path}")
    return values, mask


def _collect_statistics(
    names: Iterable[str],
    input_dir: Path,
    mask_dir: Path,
    mask_template: str,
    method: str,
) -> dict:
    count = 0
    total = None
    total_sq = None
    minimum = None
    maximum = None
    for name in names:
        values, mask = _load_map_and_mask(input_dir, mask_dir, mask_template, name)
        pixels = np.nan_to_num(values[mask], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
        if pixels.size == 0:
            continue
        count += pixels.shape[0]
        total = pixels.sum(axis=0) if total is None else total + pixels.sum(axis=0)
        if method == "zscore":
            squared = np.square(pixels).sum(axis=0)
            total_sq = squared if total_sq is None else total_sq + squared
        if method == "minmax":
            minimum = pixels.min(axis=0) if minimum is None else np.minimum(minimum, pixels.min(axis=0))
            maximum = pixels.max(axis=0) if maximum is None else np.maximum(maximum, pixels.max(axis=0))
    if count == 0:
        raise ValueError("No tissue pixels were available to calculate normalization statistics.")
    stats = {"count": np.asarray(count, dtype=np.int64)}
    if method == "zscore":
        mean = total / count
        variance = np.maximum(total_sq / count - np.square(mean), 0.0)
        stats.update(mean=mean.astype(np.float32), std=np.sqrt(variance).astype(np.float32))
    if method == "minmax":
        stats.update(minimum=minimum.astype(np.float32), maximum=maximum.astype(np.float32))
    return stats


def _normalize(values: np.ndarray, mask: np.ndarray, method: str, stats: dict,
               tissue_channels: int = 0) -> np.ndarray:
    result = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).copy()
    if method == "zscore":
        scale = np.where(stats["std"] > 1e-6, stats["std"], 1.0)
        selected = result[mask]
        selected[:, tissue_channels:] = (
            selected[:, tissue_channels:] - stats["mean"][tissue_channels:]
        ) / scale[tissue_channels:]
        result[mask] = selected
    elif method == "minmax":
        scale = np.where(stats["maximum"] > stats["minimum"], stats["maximum"] - stats["minimum"], 1.0)
        selected = result[mask]
        selected[:, tissue_channels:] = (
            selected[:, tissue_channels:] - stats["minimum"][tissue_channels:]
        ) / scale[tissue_channels:]
        result[mask] = selected
    result[~mask] = 0.0
    return result


def _crop_pad_resize(values: np.ndarray, mask: np.ndarray, image_size: int, padding: int) -> np.ndarray:
    rows, cols = np.where(mask)
    top, bottom = rows.min(), rows.max() + 1
    left, right = cols.min(), cols.max() + 1
    cropped = values[top:bottom, left:right]
    height, width, channels = cropped.shape
    side = max(height, width) + 2 * padding
    square = np.zeros((side, side, channels), dtype=np.float32)
    y = (side - height) // 2
    x = (side - width) // 2
    square[y:y + height, x:x + width] = cropped
    return cv2.resize(square, (image_size, image_size), interpolation=cv2.INTER_LINEAR)


def _geometry(mask: np.ndarray, image_size: int, padding: int) -> dict:
    """Record the exact source-to-model map transform for tile attribution."""

    rows, cols = np.where(mask)
    top, bottom = int(rows.min()), int(rows.max()) + 1
    left, right = int(cols.min()), int(cols.max()) + 1
    height, width = bottom - top, right - left
    side = max(height, width) + 2 * padding
    return {
        "source_shape": [int(mask.shape[0]), int(mask.shape[1])],
        "crop_box": [top, bottom, left, right],
        "square_side": side,
        "square_offset": [(side - height) // 2, (side - width) // 2],
        "image_size": image_size,
    }


def process_feature_maps(
    names: Iterable[str],
    input_dir: Path,
    mask_dir: Path,
    output_dir: Path,
    mask_template: str,
    method: str,
    image_size: int,
    padding: int,
    stats_path: Optional[Path],
    fit_stats: bool,
    tissue_channels: int,
    overwrite: bool,
) -> None:
    names = list(names)
    stats = {}
    if method != "initial":
        if stats_path is None:
            raise ValueError("--stats-path is required for statistical normalization.")
        if fit_stats:
            if stats_path.exists() and not overwrite:
                raise FileExistsError(f"Normalization statistics already exist: {stats_path}")
            stats = _collect_statistics(names, input_dir, mask_dir, mask_template, method)
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(stats_path, method=method, tissue_channels=tissue_channels, **stats)
        else:
            if not stats_path.is_file():
                raise FileNotFoundError(f"Fit discovery statistics first: {stats_path}")
            with np.load(stats_path) as stored:
                if str(stored["method"]) != method:
                    raise ValueError(f"Statistics in {stats_path} do not match {method}.")
                if int(stored["tissue_channels"]) != tissue_channels:
                    raise ValueError(f"Statistics in {stats_path} use a different tissue-channel count.")
                stats = {key: stored[key] for key in stored.files if key not in ("method", "tissue_channels")}
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        output_path = output_dir / name
        if output_path.exists() and not overwrite:
            continue
        values, mask = _load_map_and_mask(input_dir, mask_dir, mask_template, name)
        normalized = _normalize(values, mask, method, stats, tissue_channels)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, _crop_pad_resize(normalized, mask, image_size, padding))
        geometry_path = output_dir / "geometry" / f"{Path(name).stem}.json"
        geometry_path.parent.mkdir(parents=True, exist_ok=True)
        geometry_path.write_text(json.dumps(_geometry(mask, image_size, padding), indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize and resize MorphX feature maps.")
    parser.add_argument("--wsi-list", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument(
        "--mask-template",
        default="{stem}.svs/{stem}.svs_mask_use.png",
        help="Relative mask path with {name} and {stem} placeholders.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method", choices=("initial", "zscore", "minmax"), default="initial")
    parser.add_argument("--cohort", choices=("hcc", "luad"),
                        help="Required for zscore/minmax so tissue one-hot channels stay unchanged.")
    parser.add_argument("--stats-path", type=Path, default=None)
    parser.add_argument("--fit-stats", action="store_true", help="Fit statistics on discovery data only.")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--padding", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.method != "initial" and args.cohort is None:
        raise ValueError("--cohort is required for zscore/minmax normalization")
    process_feature_maps(
        _load_names(args.wsi_list),
        args.input_dir,
        args.mask_dir,
        args.output_dir,
        args.mask_template,
        args.method,
        args.image_size,
        args.padding,
        args.stats_path,
        args.fit_stats,
        get_cohort(args.cohort).tissue_channels if args.cohort else 0,
        args.overwrite,
    )


if __name__ == "__main__":
    main()
