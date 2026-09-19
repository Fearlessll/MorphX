import torch

from prognosis.Networks.fusion_net import FusionNet
from prognosis.Networks.resnet import resnet10


def test_hcc_and_luad_macro_dimensions():
    for channels, width in ((168, 128), (165, 256)):
        model = resnet10(
            first_covd_param=[3, 2, 1],
            input_channel_num=channels,
            output_use_sigmoid=False,
            backbone_width=width,
        )
        output = model(torch.randn(2, channels, 32, 32))
        assert output[0].shape == (2, width * 8)
        assert output[1].shape == (2, 1)


def test_fusion_dimensions():
    model = FusionNet(
        macro_input_channel_num=165,
        micro_feature_dim=165,
        macro_feature_dim=2048,
        backbone_width=256,
        output_use_sigmoid=False,
    )
    output = model(torch.randn(2, 165), torch.randn(2, 165, 32, 32))
    assert output.shape == (2, 1)
