from typing import Any, Callable

import lightning as L
import torch
import torchmetrics
from torch import nn

from seqinfer.infer.components.losses import L1RegularizationLoss
from seqinfer.utils.misc import import_object_from_path


class BaseLitModule(L.LightningModule):
    """Base lightning module"""

    def __init__(
        self,
        model: nn.Module,
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
            model (nn.Module): Pytorch module of the model.
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
        super().__init__()
        self.save_hyperparameters(ignore=["metrics", "model", "loss"])

        self.model = model
        self.loss = loss
        self.l1_loss_coef = l1_loss_coef

        self.optimizer_path = optimizer_path
        self.optimizer_kwargs = optimizer_kwargs if optimizer_kwargs else {}

        self.lr_scheduler_path = lr_scheduler_path
        self.lr_scheduler_kwargs = lr_scheduler_kwargs if lr_scheduler_kwargs else {}

        self.train_metrics = metrics.clone(prefix="train_") if metrics else None
        self.val_metrics = metrics.clone(prefix="val_") if metrics else None
        self.test_metrics = metrics.clone(prefix="test_") if metrics else None

    def get_loss(self, output: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
        """Method to compute loss"""
        loss = self.loss(output, target, **kwargs)
        if self.l1_loss_coef > 0.0:
            l1_loss = L1RegularizationLoss(lam=self.l1_loss_coef)(self.model)
            return loss + l1_loss
        return loss

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        feat, target = batch
        output = self.model(feat)
        loss = self.get_loss(output, target)
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
        loss = self.get_loss(output, target)
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


class BaseEncoderDecoder(BaseLitModule):
    """Base lightning module for encoder-decoder architectures"""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        loss: nn.Module | Callable,
        l1_loss_coef: float = 0.0,
        metrics: torchmetrics.MetricCollection | None = None,
        optimizer_path: str = "torch.optim.AdamW",
        optimizer_kwargs: dict | None = None,
        lr_scheduler_path: str | None = None,
        lr_scheduler_kwargs: dict | None = None,
    ) -> None:
        super().__init__(
            model=nn.Sequential(encoder, decoder),
            loss=loss,
            l1_loss_coef=l1_loss_coef,
            metrics=metrics,
            optimizer_path=optimizer_path,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler_path=lr_scheduler_path,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
        )
        self.encoder = encoder
        self.decoder = decoder


class BaseSeqGenDiffusion(BaseLitModule):
    """Base lightning module for diffusion models"""

    def __init__(
        self,
        model: nn.Module,
        loss: nn.Module | Callable,
        metrics: torchmetrics.MetricCollection | None = None,
        optimizer_path: str = "torch.optim.AdamW",
        optimizer_kwargs: dict | None = None,
        lr_scheduler_path: str | None = None,
        lr_scheduler_kwargs: dict | None = None,
    ) -> None:
        """Constructor method for BaseDiffusion.

        Args:
            model (nn.Module): Pytorch module for the diffusion model.
            loss (nn.Module | Callable): Pytorch module or a Callable for loss function.
            metrics (torchmetrics.MetricCollection, optional): Metrics to be computed during training
            and evaluation. Default to None.
            optimizer_path (str, optional): Path to the optimizer class. Default to "torch.optim.AdamW".
            optimizer_kwargs (dict, optional): Keyword arguments for the optimizer. Default to None.
            lr_scheduler_path (str, optional): Path to the learning rate scheduler class. Default to None.
            lr_scheduler_kwargs (dict, optional): Keyword arguments for the learning rate scheduler. Default to None.
        """
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            optimizer_path=optimizer_path,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler_path=lr_scheduler_path,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
        )

    def generate(
        self, start_sequence: torch.Tensor, num_steps: int, temperature: float = 1.0
    ) -> torch.Tensor:
        """Generate a sequence from a given start sequence.

        Args:
            start_sequence (torch.Tensor): The initial sequence to start generation from.
            num_steps (int): The number of steps to generate.
            temperature (float, optional): The temperature for sampling. Default to 1.0.

        Returns:
            torch.Tensor: The generated sequence.
        """
        generated_sequence = start_sequence.clone()
        for _ in range(num_steps):
            # Implement the sequence generation logic here
            # using self.model and the current generated_sequence
            pass

        return generated_sequence
