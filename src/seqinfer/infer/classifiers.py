from typing import Any, Callable

import lightning as L
import torch
import torchmetrics
from torch import nn

from seqinfer.utils.misc import import_object_from_path

DEFAULT_BINARY_CLASSIFICATION_METRICS = torchmetrics.MetricCollection(
    [
        torchmetrics.classification.BinaryAccuracy(),  # torchmetrics auto handle output conversion
        torchmetrics.classification.BinaryAUROC(),
        torchmetrics.classification.BinaryAveragePrecision(),
    ]
)


class LitClassifier(L.LightningModule):
    """Lightning module for general classification task"""

    def __init__(
        self,
        num_classes: int,
        model: nn.Module,
        is_output_logits: bool,
        loss: nn.Module | Callable,
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
        super().__init__()
        self.save_hyperparameters(ignore=["metrics", "model", "loss"])

        self.num_classes = num_classes
        self.model = model
        self.is_output_logits = is_output_logits
        self.loss = loss
        if self.is_output_logits:
            assert not isinstance(
                self.loss, nn.BCELoss
            ), f"{self.loss} requires model output probability density"
        else:
            assert not (
                isinstance(self.loss, (nn.BCEWithLogitsLoss, nn.CrossEntropyLoss))
            ), f"{self.loss} requires model output logits"

        self.optimizer_path = optimizer_path
        self.optimizer_kwargs = optimizer_kwargs if optimizer_kwargs else {}
        self.lr_scheduler_path = lr_scheduler_path
        self.lr_scheduler_kwargs = lr_scheduler_kwargs if lr_scheduler_kwargs else {}

        self.train_metrics = metrics.clone(prefix="train_") if metrics else None
        self.val_metrics = metrics.clone(prefix="val_") if metrics else None
        self.test_metrics = metrics.clone(prefix="test_") if metrics else None

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        feat, target = batch
        output = self.model(feat)
        loss = self.loss(output, target)
        self.log("train_loss", loss)
        if self.train_metrics is not None:
            metrics = self.train_metrics(output, target)
            self.log_dict(metrics)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        loss = self._shared_eval_step(batch, batch_idx, self.val_metrics, "val_")
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        loss = self._shared_eval_step(batch, batch_idx, self.test_metrics, "test_")
        return loss

    def _shared_eval_step(
        self,
        batch: Any,
        batch_idx: int,
        metrics: torchmetrics.MetricCollection | None,
        prefix: str,
    ) -> torch.Tensor:
        feat, target = batch
        output = self.model(feat)
        loss = self.loss(output, target)
        self.log(f"{prefix}loss", loss)
        if metrics:
            metrics.update(output, target)
        return loss

    def on_validation_epoch_end(self) -> None:
        if self.val_metrics:
            metrics = self.val_metrics.compute()
            self.log_dict(metrics)
            self.val_metrics.reset()  # reset metrics at the end of the epoch

    def on_test_epoch_end(self) -> None:
        if self.test_metrics:
            metrics = self.test_metrics.compute()
            self.log_dict(metrics)
            self.test_metrics.reset()  # reset metrics at the end of the epoch

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        feat, _ = batch
        output = self.model(feat)
        return output

    def configure_optimizers(
        self,
    ) -> (
        torch.optim.Optimizer
        | tuple[list[torch.optim.Optimizer], list[torch.optim.lr_scheduler.LRScheduler]]
    ):
        optimizer = import_object_from_path(self.optimizer_path)(
            self.model.parameters(), **self.optimizer_kwargs
        )
        if self.lr_scheduler_path:
            lr_scheduler = import_object_from_path(self.lr_scheduler_path)(
                optimizer, **self.lr_scheduler_kwargs
            )
            return [optimizer], [lr_scheduler]
        return optimizer


class LitBinaryClassifier(LitClassifier):
    """Lightning module for binary classification task"""

    def __init__(
        self,
        model: nn.Module,
        is_output_logits: bool,
        loss: nn.Module | Callable,
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
            metrics,
            optimizer_path,
            optimizer_kwargs,
            lr_scheduler_path,
            lr_scheduler_kwargs,
        )
