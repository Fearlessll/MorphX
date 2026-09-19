import pandas as pd

from prognosis.clinical import load_survival_labels


def test_hcc_tcga_labels_select_only_applicable_time_column(tmp_path):
    clinical_path = tmp_path / "TCGA.csv"
    pd.DataFrame(
        {
            "WSIs": ["TCGA-AB-1234-01A", "TCGA-CD-5678-01A"],
            "vital_status": ["Dead", "Alive"],
            "days_to_death": [300, None],
            "days_to_last_follow_up": [None, 450],
        }
    ).to_csv(clinical_path, index=False)

    events, times = load_survival_labels(
        ["TCGA-AB-1234-01A.npy", "TCGA-CD-5678-01A.npy"],
        "hcc",
        "tcga",
        clinical_csv=clinical_path,
    )

    assert events.tolist() == [1.0, 0.0]
    assert times.tolist() == [10.0, 15.0]
