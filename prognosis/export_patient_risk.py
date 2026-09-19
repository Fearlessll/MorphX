"""Export one locked OS risk score per TCGA patient for downstream endpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def export_patient_risk(predictions: pd.DataFrame) -> pd.DataFrame:
    needed = {"wsi_name", "risk_score"}
    if not needed.issubset(predictions.columns):
        raise ValueError(f"Prediction table lacks {sorted(needed - set(predictions.columns))}")
    frame = predictions.copy()
    frame["patient_barcode"] = frame["wsi_name"].astype(str).map(lambda name: Path(name).stem[:12])
    if not frame["patient_barcode"].str.match(r"^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}$").all():
        raise ValueError("This export supports TCGA barcodes only")
    if frame["patient_barcode"].duplicated().any():
        raise ValueError("Multiple WSIs share a patient barcode; an explicit aggregation rule is required")
    frame["hazard_pred"] = pd.to_numeric(frame["risk_score"], errors="coerce")
    if not np.isfinite(frame["hazard_pred"]).all():
        raise ValueError("Risk scores must be finite")
    return frame[["patient_barcode", "hazard_pred", "wsi_name"]]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Convert fixed TCGA OS predictions to patient risk table.")
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args(argv)
    result = export_patient_risk(pd.read_csv(args.predictions_csv))
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_csv, index=False)
    print(f"Exported {len(result)} unique TCGA patient risk scores")


if __name__ == "__main__":
    main()
