import json

import numpy as np
import pandas as pd
import pytest
import torch

from configs.cohorts import HCC, LUAD, default_wsi_list_path, parse_feature_families
from data_preprocess.luad_normal_ablation import candidate_mask
from prognosis.clinical import load_survival_labels
from prognosis.key_tiles import model_to_source, select_tiles
from prognosis.Networks.resnet import resnet10
from prognosis.train_mfm_msfm import _default_feature_map_dir, _median_optimal_epochs, build_parser
from prognosis.utils import cox_loss, modified_cox_loss


def test_family_ablation_uses_absolute_full_map_channels(tmp_path):
    indices, families = parse_feature_families("40+120", LUAD)
    assert families == ["texture", "nuclear"]
    assert indices == list(range(5, 165))
    with pytest.raises(FileNotFoundError):
        _default_feature_map_dir(tmp_path, LUAD, "tcga", 256)
    assert default_wsi_list_path(tmp_path, LUAD, "gd").name == "all_data_mil.json"


def test_historical_clinical_edge_cases(tmp_path):
    tcga = tmp_path / "TCGA.csv"
    pd.DataFrame({"WSIs": ["TCGA-AB-1234-01A"], "vital_status": ["Alive"],
                  "days_to_death": [np.nan], "days_to_last_follow_up": [0]}).to_csv(tcga, index=False)
    event, months = load_survival_labels(["TCGA-AB-1234-01A.npy"], HCC, "tcga", clinical_csv=tcga)
    assert event.tolist() == [0]
    assert months.tolist() == [0]

    km = tmp_path / "KMMUFH.csv"
    pd.DataFrame({"WSIs": ["123", "123"], "OS_status": [1, 1],
                  "OS": [300, -300]}).to_csv(km, index=False)
    with pytest.warns(RuntimeWarning, match="first row"):
        event, months = load_survival_labels(["123-slide.npy"], HCC, "kmmufh", clinical_csv=km)
    assert event.tolist() == [1]
    assert months.tolist() == [10]


def test_key_tile_geometry_excludes_padding_and_separates_regions():
    geometry = {"image_size": 8, "square_side": 8, "square_offset": [2, 2],
                "crop_box": [1, 5, 1, 5]}
    assert model_to_source(0, 0, geometry) is None
    contribution = np.zeros((8, 8), dtype=float)
    contribution[2, 2] = 5
    contribution[2, 3] = 4
    contribution[5, 5] = 3
    tiles = select_tiles(contribution, geometry, top_k=2, separation=1)
    assert [(row["source_row"], row["source_col"]) for row in tiles] == [(1, 1), (4, 4)]


def test_sigmoid_training_score_and_logit_inference_share_head():
    model = resnet10(first_covd_param=[3, 2, 1], input_channel_num=8,
                     backbone_width=128, output_use_sigmoid=True).eval()
    sample = torch.randn(2, 8, 32, 32)
    with torch.no_grad():
        bounded = model(sample)[1]
        model.output_use_sigmoid = False
        logit = model(sample)[1]
    assert torch.allclose(bounded, torch.sigmoid(logit))


def test_final_epoch_schedule_uses_all_ten_discovery_folds(tmp_path):
    for fold in range(10):
        output = tmp_path / f"fold_{fold}"
        output.mkdir()
        history = [{"validation": {"cindex": value}} for value in
                   ([0.6, 0.8, 0.7] if fold < 5 else [0.5, 0.6, 0.9])]
        (output / "metrics.json").write_text(
            json.dumps({"loss_function": "cox", "history": history}), encoding="utf-8"
        )
    assert _median_optimal_epochs(tmp_path, "cox") == 3
    with pytest.raises(ValueError, match="Rerun all ten discovery folds"):
        _median_optimal_epochs(tmp_path, "source_modified")


def test_training_defaults_to_standard_cox(tmp_path):
    base = ["--cohort", "hcc", "--dataset", "tcga", "--data-root", str(tmp_path)]
    parser = build_parser()
    assert parser.parse_args(base).loss_function == "cox"
    assert parser.parse_args(base + ["--loss-function", "source_modified"]).loss_function == "source_modified"


def test_historical_loss_matches_two_source_risk_set_terms():
    times = torch.tensor([1., 2., 3.])
    events = torch.tensor([1., 0., 1.])
    score = torch.tensor([0.2, 0.4, 0.8], requires_grad=True)
    risk_set = (times[None, :] >= times[:, None]).numpy()
    exp_score = np.exp(score.detach().numpy())
    row_denominator = np.log((risk_set * exp_score[None, :]).sum(axis=1))
    column_denominator = np.log((risk_set * exp_score[:, None]).sum(axis=0))
    expected = -np.mean((score.detach().numpy() - row_denominator) * events.numpy())
    expected -= np.mean((score.detach().numpy() - column_denominator) * events.numpy())
    actual = modified_cox_loss(times, events, score)
    assert actual.item() == pytest.approx(expected)
    assert actual.item() != pytest.approx(cox_loss(times, events, score).item())
    actual.backward()
    assert torch.isfinite(score.grad).all()


def test_luad_normal_candidates_follow_reviewer_definitions():
    initial = np.zeros((2, 3, 165), dtype=np.float32)
    initial[0, 0, 2] = 1  # te tumor
    initial[0, 1, 0] = 1  # necrosis
    available = np.ones((2, 3), dtype=np.uint8)
    candidate_a = candidate_mask(initial, available, "A")
    candidate_b = candidate_mask(initial, available, "B")
    assert candidate_a.tolist() == [[False, False, True], [True, True, True]]
    assert candidate_b.tolist() == [[False, True, True], [True, True, True]]
