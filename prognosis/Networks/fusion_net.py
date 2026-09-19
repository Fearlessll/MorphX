"""Macro/micro fusion network used by MSFM experiments."""

from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn as nn

from .resnet import resnet10


def _load_checkpoint_state(path: str | Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise ValueError(f"Unsupported checkpoint format: {path}")


class FusionNet(nn.Module):
    """Concatenate the macro backbone representation with micro features."""

    def __init__(
        self,
        patch_first_covd_param: Optional[Sequence[int]] = None,
        patch_input_channel_num: int = 168,
        macro_first_covd_param: Sequence[int] = (3, 2, 1),
        macro_input_channel_num: int = 168,
        micro_feature_dim: Optional[int] = None,
        macro_feature_dim: int = 1024,
        backbone_width: int = 128,
        macro_best_ckpt_path: Optional[str] = None,
        output_use_sigmoid: bool = False,
    ):
        super().__init__()
        del patch_first_covd_param, patch_input_channel_num
        if micro_feature_dim is None:
            micro_feature_dim = macro_input_channel_num
        if micro_feature_dim <= 0:
            raise ValueError("micro_feature_dim must be positive.")
        self.micro_feature_dim = int(micro_feature_dim)
        self.macro_feature_dim = int(macro_feature_dim)
        self.output_use_sigmoid = bool(output_use_sigmoid)
        if self.macro_feature_dim != backbone_width * 8:
            raise ValueError("macro_feature_dim must equal 8 * backbone_width.")

        self.macro_net = resnet10(
            first_covd_param=list(macro_first_covd_param),
            input_channel_num=int(macro_input_channel_num),
            output_use_sigmoid=False,
            backbone_width=backbone_width,
            # Keep the historical macro prediction head shape so old MFM
            # checkpoints remain loadable. The backbone representation returned
            # by ResNet is still 1024-dimensional.
            feature_dim=32,
        )
        if macro_best_ckpt_path:
            state = _load_checkpoint_state(macro_best_ckpt_path)
            missing, unexpected = self.macro_net.load_state_dict(state, strict=False)
            if missing:
                raise ValueError(
                    f"Macro checkpoint is incompatible; missing keys include {missing[:3]}"
                )
            if unexpected:
                raise ValueError(
                    f"Macro checkpoint is incompatible; unexpected keys include {unexpected[:3]}"
                )

        input_dim = self.macro_feature_dim + self.micro_feature_dim
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Dropout(0.25),
            nn.Linear(input_dim, 64),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, x_micro: torch.Tensor, x_macro: torch.Tensor) -> torch.Tensor:
        if x_micro.ndim != 2:
            raise ValueError(f"Micro features must have shape [B, D], got {tuple(x_micro.shape)}")
        if x_micro.shape[1] != self.micro_feature_dim:
            raise ValueError(
                f"Expected {self.micro_feature_dim} micro features, got {x_micro.shape[1]}"
            )
        macro_vec, _ = self.macro_net(x_macro)
        if macro_vec.shape[1] != self.macro_feature_dim:
            raise RuntimeError(
                f"Macro backbone returned {macro_vec.shape[1]} features, "
                f"expected {self.macro_feature_dim}"
            )
        features = torch.cat([macro_vec, x_micro], dim=1)
        hazard = self.classifier(features)
        return torch.sigmoid(hazard) if self.output_use_sigmoid else hazard
