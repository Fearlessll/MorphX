"""PyTorch datasets for macro feature maps and macro/micro fusion."""

from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def _feature_map_tensor(
    path: str | Path,
    channel_indices: Optional[Sequence[int]],
    image_size: Optional[int],
) -> torch.Tensor:
    array = np.asarray(np.load(path), dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"Feature map must be a 3-D HWC array, got {array.shape} from {path}")
    if channel_indices is None:
        selected = array
    else:
        if not channel_indices:
            raise ValueError("channel_indices cannot be empty.")
        max_channel = max(channel_indices)
        if max_channel >= array.shape[-1]:
            raise ValueError(
                f"{path} has {array.shape[-1]} channels but channel {max_channel} was requested."
            )
        selected = array[..., list(channel_indices)]
    selected = np.nan_to_num(selected, nan=0.0, posinf=0.0, neginf=0.0)
    tensor = torch.from_numpy(np.ascontiguousarray(selected.transpose(2, 0, 1)))
    if image_size is not None:
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=(int(image_size), int(image_size)),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    return tensor.float()


class MacroFeatureMapDataset(Dataset):
    """Load HWC feature maps and survival labels."""

    def __init__(
        self,
        data: Iterable[str | Path],
        events: Sequence[float],
        survival_times: Sequence[float],
        channel_indices: Optional[Sequence[int]] = None,
        image_size: Optional[int] = None,
    ):
        self.data = [str(path) for path in data]
        self.events = np.asarray(events, dtype=np.float32)
        self.survival_times = np.asarray(survival_times, dtype=np.float32)
        self.channel_indices = None if channel_indices is None else list(channel_indices)
        self.image_size = image_size
        if not (len(self.data) == len(self.events) == len(self.survival_times)):
            raise ValueError("Data paths, events, and survival times must have the same length.")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        feature_map = _feature_map_tensor(self.data[index], self.channel_indices, self.image_size)
        return feature_map, torch.tensor(self.survival_times[index]), torch.tensor(self.events[index])


class FusionFeatureMapDataset(MacroFeatureMapDataset):
    """Load feature maps together with one row of handcrafted micro features."""

    def __init__(
        self,
        data: Iterable[str | Path],
        events: Sequence[float],
        survival_times: Sequence[float],
        micro_features_csv: str | Path,
        micro_feature_dim: int,
        channel_indices: Optional[Sequence[int]] = None,
        image_size: Optional[int] = None,
    ):
        super().__init__(data, events, survival_times, channel_indices, image_size)
        self.micro_feature_dim = int(micro_feature_dim)
        if self.micro_feature_dim <= 0:
            raise ValueError("micro_feature_dim must be positive.")
        table = pd.read_csv(micro_features_csv)
        if "wsi_id" not in table.columns:
            raise ValueError(f"{micro_features_csv} must contain a wsi_id column.")
        id_column = "wsi_id"
        feature_columns = [column for column in table.columns if column.startswith("feature_")]
        expected_columns = [f"feature_{index}" for index in range(self.micro_feature_dim)]
        if set(feature_columns) != set(expected_columns):
            raise ValueError(
                f"{micro_features_csv} must contain exactly feature_0 through "
                f"feature_{self.micro_feature_dim - 1}."
            )
        feature_columns = expected_columns
        table["_morphx_key"] = table[id_column].astype(str).map(lambda value: Path(value).name)
        if table["_morphx_key"].duplicated().any():
            raise ValueError(f"Duplicate WSI identifiers found in {micro_features_csv}.")
        table = table.set_index("_morphx_key")
        missing = []
        rows = []
        for path in self.data:
            key = Path(path).name
            if key not in table.index and Path(key).stem in table.index:
                key = Path(key).stem
            if key not in table.index:
                missing.append(key)
                continue
            rows.append(table.loc[key, feature_columns].to_numpy(dtype=np.float32))
        if missing:
            raise KeyError(f"Missing micro features for {len(missing)} maps; examples: {missing[:5]}")
        self.micro_features = np.asarray(rows, dtype=np.float32)
        if not np.isfinite(self.micro_features).all():
            raise ValueError(f"{micro_features_csv} contains missing or nonfinite micro features")

    def __getitem__(self, index: int):
        macro, time, event = super().__getitem__(index)
        micro = torch.from_numpy(self.micro_features[index]).float()
        return (micro, macro), time, event


# Backward-compatible names for old scripts.
MyDataset = MacroFeatureMapDataset
MyFusionDataset = FusionFeatureMapDataset
