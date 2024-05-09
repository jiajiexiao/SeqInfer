from typing import Callable

import torch
import torch.nn as nn
import torchmetrics

from seqinfer.infer.components.ligntning_modules import BaseEncoderDecoder


class Autoencoder(BaseEncoderDecoder):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        loss_fn: nn.Module | Callable,
        optimizer_path: str = "torch.optim.AdamW",
        optimizer_kwargs: dict | None = None,
        lr_scheduler_path: str | None = None,
        lr_scheduler_kwargs: dict | None = None,
        l1_loss_coef: float = 0.0,
        metrics: torchmetrics.MetricCollection | None = None,
    ):
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            loss=loss_fn,
            l1_loss_coef=l1_loss_coef,
            metrics=metrics,
            optimizer_path=optimizer_path,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler_path=lr_scheduler_path,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
        )


class VAE(BaseEncoderDecoder):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        loss_fn: nn.Module | Callable,
        optimizer_path: str = "torch.optim.AdamW",
        optimizer_kwargs: dict | None = None,
        lr_scheduler_path: str | None = None,
        lr_scheduler_kwargs: dict | None = None,
        l1_loss_coef: float = 0.0,
        metrics: torchmetrics.MetricCollection | None = None,
    ):
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            loss=loss_fn,
            l1_loss_coef=l1_loss_coef,
            metrics=metrics,
            optimizer_path=optimizer_path,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler_path=lr_scheduler_path,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class Seq2Seq(BaseEncoderDecoder):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        loss_fn: nn.Module | Callable,
        optimizer_path: str = "torch.optim.AdamW",
        optimizer_kwargs: dict | None = None,
        lr_scheduler_path: str | None = None,
        lr_scheduler_kwargs: dict | None = None,
        l1_loss_coef: float = 0.0,
        metrics: torchmetrics.MetricCollection | None = None,
    ):
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            loss=loss_fn,
            l1_loss_coef=l1_loss_coef,
            metrics=metrics,
            optimizer_path=optimizer_path,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler_path=lr_scheduler_path,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
        )

    def forward(
        self, src: torch.Tensor, tgt: torch.Tensor, teacher_forcing_ratio: float = 0.5
    ) -> torch.Tensor:
        encoder_output, hidden = self.encoder(src)
        decoded_output = self.decoder(tgt, encoder_output, hidden, teacher_forcing_ratio)
        return decoded_output
