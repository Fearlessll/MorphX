import numpy as np
import pandas as pd
import pytest

from prognosis.endpoints import (
    EARLY_DEATH,
    EXCLUDED_CENSORED,
    EXCLUDED_INVALID,
    SURVIVED_BEYOND,
    assign_three_year_os_groups,
    fixed_os_risk_pfi_analysis,
    _read_table,
)
from configs.cohorts import get_cohort


def test_cohort_specs_lock_channel_slices_and_external_roles():
    hcc = get_cohort("HCC")
    luad = get_cohort("luad")

    assert hcc.total_channels == 168
    assert (hcc.family_slices["tissue"].start, hcc.family_slices["tissue"].stop) == (0, 8)
    assert (hcc.family_slices["texture"].start, hcc.family_slices["texture"].stop) == (8, 48)
    assert (hcc.family_slices["nuclear"].start, hcc.family_slices["nuclear"].stop) == (48, 168)
    assert luad.total_channels == 165
    assert (luad.family_slices["tissue"].start, luad.family_slices["tissue"].stop) == (0, 5)
    assert hcc.discovery_dataset == "tcga"
    assert hcc.external_dataset == "kmmufh"
    assert luad.external_dataset == "gd"

    with pytest.raises(ValueError):
        hcc.validate_training_dataset("kmmufh")
    with pytest.raises(ValueError):
        luad.validate_external_dataset("tcga")


def test_three_year_os_boundary_and_event_encodings():
    frame = pd.DataFrame(
        {
            "os_time_months": [36, 36, 36.1, 12, np.nan, 0],
            "os_status": [1, 0, "Dead", "Alive", 1, 0],
        }
    )

    labels = assign_three_year_os_groups(frame)

    assert labels.tolist() == [
        EARLY_DEATH,
        EXCLUDED_CENSORED,
        SURVIVED_BEYOND,
        EXCLUDED_CENSORED,
        EXCLUDED_INVALID,
        EXCLUDED_INVALID,
    ]


def test_fixed_pfi_risk_uses_complete_os_risk_distribution_for_cutoff():
    frame = pd.DataFrame(
        {
            "patient_barcode": ["P1", "P2", "P3", "P4", "P5"],
            "hazard_pred": [0.0, 1.0, 2.0, 3.0, 4.0],
            "PFI": [0, 1, 0, 1, np.nan],
            "PFI.time": [100, 80, 60, 40, np.nan],
        }
    )

    patients, summary = fixed_os_risk_pfi_analysis(frame)

    assert summary["matched_valid_pfi_n"] == 4
    assert summary["unmatched_or_invalid_pfi_n"] == 1
    assert summary["km_cutoff_hazard_pred"] == 2.0
    assert summary["km_low_n"] == 3
    assert summary["km_high_n"] == 1
    assert patients["hazard_pred_z"].iloc[0] < patients["hazard_pred_z"].iloc[1]
    assert "retraining" in summary["analysis_scope"]


def test_pfi_endpoint_can_read_the_tcga_cdr_excel_sheet(tmp_path):
    path = tmp_path / "endpoints.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({"bcr_patient_barcode": ["P1"], "PFI": [1], "PFI.time": [100]}).to_excel(
            writer, sheet_name="TCGA-CDR", index=False
        )
    frame = _read_table(path, "TCGA-CDR")
    assert frame["PFI.time"].tolist() == [100]
