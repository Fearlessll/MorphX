import math

import torch

from prognosis.utils import cox_loss


def test_cox_loss_uses_breslow_ties():
    times = torch.tensor([2.0, 2.0, 1.0])
    events = torch.tensor([1.0, 1.0, 0.0])
    scores = torch.tensor([[0.0], [math.log(2.0)], [0.0]])

    expected = (2.0 * math.log(3.0) - math.log(2.0)) / 2.0
    assert torch.isclose(cox_loss(times, events, scores), torch.tensor(expected), atol=1e-6)
