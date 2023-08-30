import torch
from torch import nn


class L1RegularizationLoss(nn.Module):
    """L1 regularization loss module that adds L1 loss on the weights (biases are excluded from
    the calculation) of a model to induce sparsity."""

    def __init__(self, weight_decay: float = 0.01) -> None:
        """Initialization

        Args:
            weight_decay (float, optional): L1 penalty coefficient. Defaults to 0.01.
        """
        super().__init__()
        self.weight_decay = weight_decay

    def forward(self, model: nn.Module) -> torch.Tensor:
        """Computes the L1 loss on the weights of the model."""
        l1_loss = 0.0
        num_parameters = 0
        for name, param in model.named_parameters():
            if "bias" not in name and param.requires_grad:
                l1_loss += torch.sum(torch.abs(param))
                num_parameters += param.numel()
        return self.weight_decay * l1_loss / num_parameters
