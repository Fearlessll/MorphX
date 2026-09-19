import numpy as np
import pandas as pd
import pytest
import torch

from prognosis.data_loaders import FusionFeatureMapDataset, MacroFeatureMapDataset


def test_macro_and_fusion_datasets(tmp_path):
    feature_path = tmp_path / "sample.npy"
    values = np.zeros((8, 8, 5), dtype=np.float32)
    values[0, 0, 0] = np.nan
    np.save(feature_path, values)
    events = np.asarray([1], dtype=np.float32)
    times = np.asarray([12], dtype=np.float32)

    macro = MacroFeatureMapDataset([feature_path], events, times, [0, 2, 4], image_size=4)
    macro_value, macro_time, macro_event = macro[0]
    assert macro_value.shape == (3, 4, 4)
    assert torch.isfinite(macro_value).all()
    assert macro_time.item() == 12
    assert macro_event.item() == 1

    csv_path = tmp_path / "micro.csv"
    pd.DataFrame({"wsi_id": [feature_path.name], "feature_0": [1], "feature_1": [2]}).to_csv(
        csv_path, index=False
    )
    fusion = FusionFeatureMapDataset(
        [feature_path], events, times, csv_path, micro_feature_dim=2, image_size=4
    )
    (micro, feature), _, _ = fusion[0]
    assert micro.shape == (2,)
    assert torch.isfinite(micro).all()
    assert feature.shape == (5, 4, 4)

    pd.DataFrame({"wsi_id": [feature_path.name], "feature_0": [1], "feature_1": [np.nan]}).to_csv(
        csv_path, index=False
    )
    with pytest.raises(ValueError, match="missing or nonfinite"):
        FusionFeatureMapDataset([feature_path], events, times, csv_path, micro_feature_dim=2)
