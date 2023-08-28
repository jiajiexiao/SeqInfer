from typing import Callable

import torch
import torchmetrics
from torch import nn

from seqinfer.infer.components.ligntning_modules import BaseLitModule

DEFAULT_BINARY_CLASSIFICATION_METRICS = torchmetrics.MetricCollection(
    [
        torchmetrics.classification.BinaryAccuracy(),  # torchmetrics auto handle output conversion
        torchmetrics.classification.BinaryAUROC(),
        # torchmetrics.classification.BinaryAveragePrecision(), # this is commented as it requires
        # target to be int or torch.Long dtype, while some loss such as torch.nn.BCEWithLogitsLoss
        # requires float target.
    ]
)


class LitClassifier(BaseLitModule):
    """Lightning module for general classification task"""

    def __init__(
        self,
        num_classes: int,
        model: nn.Module,
        is_output_logits: bool,
        loss: nn.Module | Callable,
        l1_loss_coef: float = 0.0,
        metrics: torchmetrics.MetricCollection | None = None,
        optimizer_path: str = "torch.optim.AdamW",
        optimizer_kwargs: dict | None = None,
        lr_scheduler_path: str | None = None,
        lr_scheduler_kwargs: dict | None = None,
    ) -> None:
        """Constructor method for LitClassifier.

        Args:
            num_classes (int): number of classes to classify.
            model (nn.Module): Pytorch module of the model.
            is_output_logits (bool): whether the model output raw logits or not.
            loss (nn.Module | Callable): Pytorch module or a Callable for loss function.
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
        self.num_classes = num_classes
        self.is_output_logits = is_output_logits
        if self.is_output_logits:
            assert not isinstance(
                self.loss, nn.BCELoss
            ), f"{self.loss} requires model output probability density"
        else:
            assert not (
                isinstance(self.loss, (nn.BCEWithLogitsLoss, nn.CrossEntropyLoss))
            ), f"{self.loss} requires model output logits"

    def predict_prob(self, feat: torch.Tensor) -> torch.Tensor:
        """Method to output probability for predicted class(es)

        Args:
            feat (torch.Tensor): input feature tensor

        Returns:
            torch.Tensor: probability for predicted class(es)
        """
        output = self.model(feat)
        if self.is_output_logits:
            shape = output.shape
            if len(shape) > 1 and output.shape[1] > 1:
                # output probability for all predicted classes
                output_prob = nn.functional.softmax(output, dim=1)
            else:
                # only output probability for positive class in binary classification
                output_prob = nn.functional.sigmoid(output)
            return output_prob
        else:  # output is already the probability for predicted class
            return output


class LitBinaryClassifier(LitClassifier):
    """Lightning module for binary classification task"""

    def __init__(
        self,
        model: nn.Module,
        is_output_logits: bool,
        loss: nn.Module | Callable,
        l1_loss_coef: float = 0.0,
        metrics: torchmetrics.MetricCollection | None = DEFAULT_BINARY_CLASSIFICATION_METRICS,
        optimizer_path: str = "torch.optim.AdamW",
        optimizer_kwargs: dict | None = None,
        lr_scheduler_path: str | None = None,
        lr_scheduler_kwargs: dict | None = None,
    ) -> None:
        super().__init__(
            2,
            model,
            is_output_logits,
            loss,
            l1_loss_coef,
            metrics,
            optimizer_path,
            optimizer_kwargs,
            lr_scheduler_path,
            lr_scheduler_kwargs,
        )
