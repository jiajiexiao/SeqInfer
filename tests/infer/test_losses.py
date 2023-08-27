import torch
from torch import nn

from seqinfer.infer.components import losses


class TestL1RegularizationLoss:
    """Unit test class for L1RegularizationLoss"""

    def test_l1_reg_loss(self):
        """Test conventional use case"""
        model = nn.Linear(10, 1)
        l1_reg = losses.L1RegularizationLoss(weight_decay=0.1)
        with torch.no_grad():
            l1_loss = l1_reg(model)
            torch.testing.assert_close(
                l1_loss.item(), 0.1 * torch.sum(torch.abs(model.weight)).item()
            )

    def test_zero_weight_decay(self):
        """Test if weight decay = 0 return 0 loss"""
        model = nn.Linear(10, 1)
        l1_reg = losses.L1RegularizationLoss(weight_decay=0.0)
        with torch.no_grad():
            l1_loss = l1_reg(model)
            torch.testing.assert_close(l1_loss.item(), 0.0)

    def test_exlude_bias(self):
        """Test if bias term is excluded"""
        model = nn.Linear(10, 1)
        model.weight.data.fill_(0)  # set weight = 0
        model.bias.data.fill_(1)  # set bias = 1
        l1_reg = losses.L1RegularizationLoss(weight_decay=0.1)
        with torch.no_grad():
            l1_loss = l1_reg(model)
            torch.testing.assert_close(l1_loss.item(), 0.0)  # Bias not included
