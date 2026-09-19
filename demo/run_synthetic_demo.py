"""Run a small, CPU-only MorphX end-to-end smoke test."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _make_feature_maps(
    directory: Path,
    names: list[str],
    channels: int,
    image_size: int,
    rng: np.random.Generator,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for index, name in enumerate(names):
        signal = (index - len(names) / 2.0) / max(len(names), 1)
        array = rng.normal(0.0, 1.0, (image_size, image_size, channels)).astype(np.float32)
        array[..., 0] += np.float32(signal)
        np.save(directory / name, array)
        geometry = {"source_shape": [image_size, image_size],
                    "crop_box": [0, image_size, 0, image_size],
                    "square_side": image_size, "square_offset": [0, 0],
                    "image_size": image_size}
        _write_json(directory / "geometry" / f"{Path(name).stem}.json", geometry)


def _make_mock_micro_features(directory: Path, names: list[str]) -> Path:
    """Exercise the fusion interface; these are synthetic, not Pathomics features."""

    output = directory / "fold_0.csv"
    rows = []
    for name in names:
        values = np.load(directory / name).mean(axis=(0, 1))
        rows.append({"wsi_id": name, **{f"feature_{index}": float(value)
                                        for index, value in enumerate(values)}})
    pd.DataFrame(rows).to_csv(output, index=False)
    return output


def _clinical_rows(names: list[str], external: bool, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for index, name in enumerate(names):
        stem = Path(name).stem
        clinical_id = stem.split("-", 1)[0] if external else stem
        event = int(index % 2 == 0)
        months = float(8.0 + (index + 1) * 1.7 + rng.uniform(0.0, 2.0))
        if external:
            rows.append({"WSIs": clinical_id, "OS_status": event, "OS": months * 30.0})
        else:
            rows.append(
                {
                    "WSIs": clinical_id,
                    "vital_status": "Dead" if event else "Alive",
                    "days_to_death": months * 30.0 if event else np.nan,
                    "days_to_last_follow_up": months * 30.0 if not event else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _build_demo_data(data_root: Path, image_size: int, seed: int) -> dict[str, Path]:
    rng = np.random.default_rng(seed)
    tcga_dir = data_root / "TCGA"
    external_dir = data_root / "KMMUFH"
    train_names = [f"TCGA-SY-{index:04d}-01A.npy" for index in range(1, 13)]
    validation_names = [f"TCGA-SY-{index:04d}-01A.npy" for index in range(13, 21)]
    external_names = [f"DEMO{index:03d}-slide.npy" for index in range(1, 9)]

    _make_feature_maps(tcga_dir / "feature_maps", train_names + validation_names, 168, image_size, rng)
    _make_feature_maps(external_dir / "feature_maps", external_names, 168, image_size, rng)
    _clinical_rows(train_names + validation_names, False, rng).to_csv(tcga_dir / "TCGA.csv", index=False)
    _clinical_rows(external_names, True, rng).to_csv(
        external_dir / "KMMUFH_data_label_fix.csv", index=False
    )
    _write_json(
        tcga_dir / "split_data_fold_0.json",
        {"train_data": train_names, "test_data": validation_names},
    )
    _write_json(tcga_dir / "all_data.json", train_names + validation_names)
    _write_json(tcga_dir / "one_wsi.json", train_names[:1])
    _write_json(external_dir / "all_data.json", external_names)
    _make_mock_micro_features(tcga_dir / "feature_maps", train_names + validation_names)
    _make_mock_micro_features(external_dir / "feature_maps", external_names)
    return {
        "data_root": data_root,
        "tcga_feature_maps": tcga_dir / "feature_maps",
        "external_feature_maps": external_dir / "feature_maps",
        "tcga_clinical": tcga_dir / "TCGA.csv",
        "external_clinical": external_dir / "KMMUFH_data_label_fix.csv",
        "split_dir": tcga_dir,
        "external_wsi_list": external_dir / "all_data.json",
        "one_wsi_list": tcga_dir / "one_wsi.json",
    }


def _run(command: list[str]) -> None:
    print("$ " + " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the MorphX synthetic CPU demo.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output",
        help="Directory for generated data, checkpoints, and metrics.",
    )
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.image_size < 32:
        raise ValueError("--image-size must be at least 32 for the bundled ResNet backbone.")
    output_dir = args.output_dir.resolve()
    data = _build_demo_data(output_dir / "data", args.image_size, args.seed)
    train_output = output_dir / "train"
    external_output = output_dir / "external"

    _run(
        [
            sys.executable,
            "-m",
            "prognosis.train_mfm_msfm",
            "--cohort",
            "hcc",
            "--dataset",
            "tcga",
            "--data-root",
            str(data["data_root"]),
            "--clinical-csv",
            str(data["tcga_clinical"]),
            "--feature-map-dir",
            str(data["tcga_feature_maps"]),
            "--splits-dir",
            str(data["split_dir"]),
            "--feature-families",
            "all",
            "--model",
            "macro",
            "--image-size",
            str(args.image_size),
            "--folds",
            "0",
            "--epochs",
            "1",
            "--batch-size",
            "0",
            "--workers",
            "0",
            "--device",
            "cpu",
            "--seed",
            str(args.seed),
            "--output-dir",
            str(train_output),
        ]
    )
    checkpoint = train_output / "fold_0" / "best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Training did not create the expected checkpoint: {checkpoint}")

    _run([
        sys.executable, "-m", "prognosis.key_tiles", "--cohort", "hcc",
        "--checkpoint", str(checkpoint), "--wsi-list", str(data["one_wsi_list"]),
        "--feature-map-dir", str(data["tcga_feature_maps"]),
        "--geometry-dir", str(data["tcga_feature_maps"] / "geometry"),
        "--output-csv", str(output_dir / "key_tiles.csv"), "--steps", "2",
        "--top-k", "4", "--device", "cpu",
    ])

    _run(
        [
            sys.executable,
            "-m",
            "prognosis.evaluate_external",
            "--cohort",
            "hcc",
            "--dataset",
            "kmmufh",
            "--data-root",
            str(data["data_root"]),
            "--checkpoint",
            str(checkpoint),
            "--wsi-list",
            str(data["external_wsi_list"]),
            "--clinical-csv",
            str(data["external_clinical"]),
            "--feature-map-dir",
            str(data["external_feature_maps"]),
            "--feature-families",
            "all",
            "--image-size",
            str(args.image_size),
            "--batch-size",
            "4",
            "--workers",
            "0",
            "--device",
            "cpu",
            "--output-dir",
            str(external_output),
        ]
    )
    metrics_path = external_output / "external_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    fusion_output = output_dir / "fusion"
    _run([
        sys.executable, "-m", "prognosis.train_mfm_msfm", "--cohort", "hcc",
        "--dataset", "tcga", "--data-root", str(data["data_root"]),
        "--feature-map-dir", str(data["tcga_feature_maps"]),
        "--splits-dir", str(data["split_dir"]), "--model", "fusion",
        "--micro-feature-template", str(data["tcga_feature_maps"] / "fold_{fold}.csv"),
        "--micro-feature-dim", "168", "--macro-checkpoint-template",
        str(train_output / "fold_{fold}" / "best.pt"), "--image-size", str(args.image_size),
        "--folds", "0", "--epochs", "1", "--batch-size", "0", "--workers", "0",
        "--device", "cpu", "--seed", str(args.seed), "--output-dir", str(fusion_output),
    ])
    _run([
        sys.executable, "-m", "prognosis.evaluate_external", "--cohort", "hcc",
        "--dataset", "kmmufh", "--data-root", str(data["data_root"]),
        "--checkpoint", str(fusion_output / "fold_0" / "best.pt"),
        "--feature-map-dir", str(data["external_feature_maps"]), "--model", "fusion",
        "--micro-features-csv", str(data["external_feature_maps"] / "fold_0.csv"),
        "--micro-feature-dim", "168", "--image-size", str(args.image_size),
        "--device", "cpu", "--output-dir", str(output_dir / "fusion_external"),
    ])
    _run([
        sys.executable, "-m", "prognosis.train_mfm_msfm", "--cohort", "hcc",
        "--dataset", "tcga", "--data-root", str(data["data_root"]),
        "--feature-map-dir", str(data["tcga_feature_maps"]),
        "--stage", "final", "--epochs", "1", "--image-size", str(args.image_size),
        "--batch-size", "0", "--device", "cpu", "--output-dir", str(output_dir / "final_macro"),
    ])
    print("\nSynthetic demo completed.")
    print(json.dumps({"checkpoint": str(checkpoint), "external_metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
