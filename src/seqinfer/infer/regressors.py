from typing import Callable

import torchmetrics
from torch import nn

from seqinfer.infer.components.ligntning_modules import BaseLitModule


class LitRegressor(BaseLitModule):
    """Lightning module for general regression tasks.

    This class is inherited from BaseLitModule and serves as a placeholder for regression tasks.
    Currently the only change is to have the nn.MSELoss as the default loss.
    """

    def __init__(
        self,
        model: nn.Module,
        loss: nn.Module | Callable = nn.MSELoss(),
        l1_loss_coef: float = 0.0,
        metrics: torchmetrics.MetricCollection | None = None,
        optimizer_path: str = "torch.optim.AdamW",
        optimizer_kwargs: dict | None = None,
        lr_scheduler_path: str | None = None,
        lr_scheduler_kwargs: dict | None = None,
    ) -> None:
        """Constructor method for LitRegressor.

        Args:
            model (nn.Module): Pytorch module of the model.
            loss (nn.Module | Callable): Pytorch module or a Callable for loss function. Default to
            torch's MSELoss().
            l1_loss_coef: (float, optional): Coef for l1 loss added to the total loss for sparse
            weights and biases of the network. Default to 0.0, meaning no l1 sparsity is needed.
            metrics (torchmetrics.MetricCollection, optional): A MetricCollection of evaluation
            metrics. Defaults to None.
            optimizer_path (str, optional): import path of the optimizer. Defaults to
            "torch.optim.AdamW".
            optimizer_kwargs (dict | None, optional): kwargs for the optimizer. Defaults to None.
            lr_scheduler_path (str | None, optional):  import path of the learning rate scheduler.
            Defaults to None.
            lr_scheduler_kwargs (dict | None, optional): kwargs for the learning rate scheduler.
            Defaults to None.
        """
        super().__init__(
            model,
            loss,
            l1_loss_coef,
            metrics,
            optimizer_path,
            optimizer_kwargs,
            lr_scheduler_path,
            lr_scheduler_kwargs,
        )
