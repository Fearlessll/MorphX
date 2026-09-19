"""Create audited LUAD candidate-normal maps for the reviewer ablation.

Candidate A is available tissue outside all existing tissue classes. Candidate
B is available tissue outside the LUAD ``te`` tumor channel. These masks are
applied spatially to the same 165-channel processed maps used by MFM.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from configs.cohorts import LUAD


def candidate_mask(initial_map: np.ndarray, available_mask: np.ndarray,
                   candidate: str, available_threshold: float = 0,
                   tissue_threshold: float = 0.5) -> np.ndarray:
    if initial_map.ndim != 3 or initial_map.shape[-1] < LUAD.tissue_channels:
        raise ValueError("Initial LUAD map must contain the five canonical tissue channels")
    height, width = initial_map.shape[:2]
    available = np.asarray(Image.fromarray(available_mask.astype(np.uint8)).resize(
        (width, height), resample=Image.Resampling.NEAREST
    ))
    available = available > available_threshold
    tissue = np.asarray(initial_map[..., :LUAD.tissue_channels], dtype=np.float32)
    if candidate == "A":
        occupied = np.nanmax(tissue, axis=-1) > tissue_threshold
    elif candidate == "B":
        occupied = tissue[..., LUAD.tissue_labels.index("te")] > tissue_threshold
    else:
        raise ValueError("candidate must be A or B")
    return available & ~occupied


def run(args) -> pd.DataFrame:
    with args.wsi_list.open(encoding="utf-8-sig") as handle:
        names = json.load(handle)
    if not isinstance(names, list) or not names or len(set(names)) != len(names):
        raise ValueError("--wsi-list must contain unique names")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    qc = []
    for name in names:
        stem = Path(name).stem
        initial_path = args.initial_feature_dir / name
        final_path = args.final_feature_dir / name
        mask_path = args.available_mask_dir / args.available_template.format(stem=stem, name=name)
        initial = np.load(initial_path, mmap_mode="r")
        final = np.load(final_path, mmap_mode="r")
        if not mask_path.is_file():
            raise FileNotFoundError(f"Cannot read available tissue mask: {mask_path}")
        available = np.asarray(Image.open(mask_path).convert("L"))
        if final.ndim != 3 or final.shape[-1] != LUAD.total_channels:
            raise ValueError(f"Expected 165-channel final LUAD map: {final_path}")
        selected_initial = candidate_mask(initial, available, args.candidate,
                                          args.available_threshold, args.tissue_threshold)
        selected_final = np.asarray(Image.fromarray(selected_initial.astype(np.uint8)).resize(
            (final.shape[1], final.shape[0]), resample=Image.Resampling.NEAREST
        )).astype(bool)
        count = int(selected_final.sum())
        if count == 0 and not args.allow_empty:
            raise ValueError(f"Candidate {args.candidate} contains no pixels for {name}; inspect QC")
        result = np.asarray(final, dtype=np.float32).copy()
        result[~selected_final] = 0
        result = np.nan_to_num(result, nan=0, posinf=0, neginf=0)
        np.save(args.output_dir / name, result)
        qc.append({"wsi_id": name, "candidate": args.candidate, "initial_height": initial.shape[0],
                   "initial_width": initial.shape[1], "available_height": available.shape[0],
                   "available_width": available.shape[1], "final_height": final.shape[0],
                   "final_width": final.shape[1], "retained_final_pixels": count,
                   "retained_fraction": float(count / selected_final.size),
                   "available_to_initial_scale_y": float(available.shape[0] / initial.shape[0]),
                   "available_to_initial_scale_x": float(available.shape[1] / initial.shape[1]),
                   "initial_to_final_scale_y": float(final.shape[0] / initial.shape[0]),
                   "initial_to_final_scale_x": float(final.shape[1] / initial.shape[1])})
    table = pd.DataFrame(qc)
    table.to_csv(args.output_dir / "mask_qc.csv", index=False)
    return table


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Prepare LUAD candidate-normal MFM ablation maps.")
    parser.add_argument("--candidate", choices=("A", "B"), required=True)
    parser.add_argument("--wsi-list", type=Path, required=True)
    parser.add_argument("--initial-feature-dir", type=Path, required=True)
    parser.add_argument("--available-mask-dir", type=Path, required=True)
    parser.add_argument("--available-template", default="{stem}.png")
    parser.add_argument("--final-feature-dir", type=Path, required=True)
    parser.add_argument("--available-threshold", type=float, default=0)
    parser.add_argument("--tissue-threshold", type=float, default=0.5)
    parser.add_argument("--allow-empty", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    table = run(args)
    print(f"Prepared {len(table)} masked maps; empty masks: "
          f"{int((table['retained_final_pixels'] == 0).sum())}")


if __name__ == "__main__":
    main()
