from typing import Type

import torch
from torch import nn

from seqinfer.utils.misc import import_object_from_path


class FNN(nn.Module):
    """Feedforward neural network with options to customize the following model configs:

    - fully connected network's architecture
    - the choice of activation
    - whether use batchnorm or not and where to put bn relative to the activation function
    - the use of dropout
    - whether to squeeze the output when there is 1d output
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: list[int],
        p_dropout: float | None = 0.5,
        use_batchnorm: str | bool = "after",
        activation_function: str | nn.Module | None = nn.ReLU,
        squeeze_output: bool = False,
    ) -> None:
        """Initialize FNN class

        Args:
            input_dim (int): Input dimension.
            output_dim (int): Output dimension.
            hidden_dim (list[int]): List of hidden layer dimensions.
            p_dropout (float | None, optional): Dropout probability. Default to 0.5.
            use_batchnorm (str | bool, optional):
                Whether to use batch normalization 'before'/'after' the activation or False to
                not use bn at all. Default to 'after'.
            activation_function (str | nn.Module | None, optional):
                Activation function module or its path or None. When it's the path, the
                corresponding module will be imported; when it's None, `nn.Identity`
                will be used. Default to `nn.ReLU`.
            squeeze_output (bool, optional):
                Whether to squeeze the last dimension of the output tensor. Having this option is to
                squeeze the last dim if that dim of x is 1. This operation can be useful when the
                criterion/loss functions require the input and target have the same shape. For
                example, the target can have a shape of (N, ) while the unsqueezed output from this
                FNN is (N, 1). This will not be accepted by loss function like
                `nn.BCEWithLogitsLoss` and squeeze (N, 1) to (N, ) is needed. Defaults to False.
        """
        super().__init__()

        if use_batchnorm in ["before", "after", False]:
            self.use_batchnorm = use_batchnorm
        else:
            raise ValueError(
                f"use_batchnorm can be only `before`, `after`, `False` but got {use_batchnorm}"
            )

        if isinstance(activation_function, str):
            activation = import_object_from_path(activation_function)
        elif activation_function is None:
            activation = import_object_from_path("nn.Identity")
        else:
            activation = activation_function

        self.squeeze_output = squeeze_output

        n_units_per_layer = [input_dim] + hidden_dim
        self.layers = nn.ModuleList()
        for i in range(len(n_units_per_layer) - 1):
            modules = nn.Sequential(nn.Linear(n_units_per_layer[i], n_units_per_layer[i + 1]))

            if self.use_batchnorm == "before":
                modules.append(nn.BatchNorm1d(num_features=n_units_per_layer[i + 1]))

            modules.append(activation())
            if self.use_batchnorm == "after":
                modules.append(nn.BatchNorm1d(num_features=n_units_per_layer[i + 1]))

            if p_dropout is not None:
                modules.append(nn.Dropout(p_dropout))

            self.layers.append(modules)

        self.layers.append(nn.Linear(n_units_per_layer[-1], output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        if self.squeeze_output:
            x = torch.squeeze(x, dim=-1)
        return x


class EnhancedTransformerEncoderLayer(nn.Module):
    """Enhanced version of torch's TransformerEncoderLayer module with the following customization.

    - Add access to attention weights via (dynamically) setting `self.output_attn_weights = True`.
      This can be helpful to manually check attention maps in inference time but disable it in
      training time.
    - Allow choice of w./w.o. residual and pooling.
    - Support customize feed-forward network.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        residual: bool = True,
        norm_first: bool = False,
        pooling: str | bool = False,
        p_dropout: float = 0.5,
        fnn: str | Type[nn.Module] = FNN,
        fnn_kwargs: dict | None = None,
        batch_first: bool = True,
        other_multihead_attn_kwargs: dict | None = None,
        output_attn_weights: bool = False,
    ):
        super().__init__()

        if other_multihead_attn_kwargs is None:
            other_multihead_attn_kwargs = {}

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=batch_first,
            dropout=p_dropout,
            **other_multihead_attn_kwargs,
        )

        if fnn_kwargs:
            assert (
                "output_dim" in fnn_kwargs
            ), "Key of `output_dim` is required in fnn_kwargs to check if it equals embed_dim."

            if residual and fnn_kwargs["output_dim"] != embed_dim:
                raise ValueError(
                    "residual can't be added when the FFN's output_dim is deferent from the "
                    "embed_dim."
                )
            self.fnn_kwargs = fnn_kwargs

        else:  # if fnn_kwargs uses the default value
            self.fnn_kwargs = dict(
                input_dim=embed_dim,
                output_dim=embed_dim,
                hidden_dim=[embed_dim],
                use_batchnorm=False,
                activation_function="nn.ReLU",
                p_dropout=p_dropout,
            )
        self.feed_forward = (
            import_object_from_path(fnn)(**self.fnn_kwargs)
            if isinstance(fnn, str)
            else fnn(**self.fnn_kwargs)
        )
        self.pooling = pooling
        self.residual = residual
        self.norm_first = norm_first
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(p_dropout)
        self.dropout2 = nn.Dropout(p_dropout)
        self.output_attn_weights = output_attn_weights

    def forward(
        self, x: torch.Tensor, **kwargs: dict
    ) -> tuple[torch.Tensor, torch.Tensor | torch.Tensor]:
        # In current version of pytorch, nn.MultiheadAttention.forward has an argument called
        # `attn_mask` but nn.TransformerEncoder.forward uses `src_mask` for the same argument.
        # Similarly for `src_key_padding_mask`, `mask`. To avoid unexpected keyword argument,
        # rename the key name as below:
        if "src_mask" in kwargs:
            kwargs["attn_mask"] = kwargs.pop("src_mask")

        if "mask" in kwargs:
            kwargs["attn_mask"] = kwargs.pop("mask")

        if "src_key_padding_mask" in kwargs:
            kwargs["key_padding_mask"] = kwargs.pop("src_key_padding_mask")

        if self.norm_first:
            # Check "On Layer Normalization in the Transformer Architecture" to know about pre-LN
            # transformer https://arxiv.org/abs/2002.04745. Particularly, the following code
            # implements the post- and pre-LN as shown in fig 1.
            x_norm = self.layer_norm1(x)
            x_attn, attn_output_weights = self.multihead_attn(x_norm, x_norm, x_norm, **kwargs)

            x_out = self.dropout1(x_attn) + x if self.residual else self.dropout1(x_attn)
            x_ff = self.dropout2(self.feed_forward(self.layer_norm2(x_out)))
            x_out = x_ff + x_out if self.residual else x_ff
        else:
            x_attn, attn_output_weights = self.multihead_attn(x, x, x, **kwargs)
            x_out = self.dropout1(x_attn) + x if self.residual else self.dropout1(x_attn)
            x_norm = self.layer_norm1(x_out)
            x_ff = self.dropout2(self.feed_forward(x_norm))
            x_out = x_ff + x_norm if self.residual else x_ff
            x_out = self.layer_norm2(x_out)

        if self.pooling == "mean":
            x_out = torch.mean(x_out, dim=1)  # average pooling over features
        elif self.pooling == "max":
            x_out = torch.max(x_out, dim=1)
        elif self.pooling is False:
            pass
        else:
            raise NotImplementedError(
                f"{self.pooling} is not supported yet. Only accept `mean`,`max` or `False` value for now."
            )

        if self.output_attn_weights:
            return x_out, attn_output_weights
        return x_out


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding as described in 'Attention is all you need'."""

    def __init__(
        self, embed_dim: int, dropout: float = 0.0, max_len: int = 5000, batch_first: bool = False
    ):
        """
        Args:
            embed_dim (int): embed_dim of input, i.e. d_model for transformer's encoder.
            dropout (float): dropout rate. Default: 0.0.
            max_len (int): maximum sequence length. Default: 5000.
            batch_first (bool): whether the 0th dim of the input is the batch dim. Default: False

        Example usage:
            x = torch.randn(100, 32, 30)
            x_batch_first = x.transpose(0, 1)
            SinusoidalPositionalEncoding(embed_dim=30, max_len=500, batch_first=False)(x)
            SinusoidalPositionalEncoding(embed_dim=30, max_len=500, batch_first=True)(x_batch_first)
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first
        self.max_len = max_len
        positions = torch.arange(self.max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-torch.log(10000.0) / embed_dim))

        positional_embeddings = torch.zeros(
            self.max_len, 1, embed_dim
        )  # Default batch_first=False, i.e. input with (seq_len, batch_size, embed_dim)
        positional_embeddings[:, 0, 0::2] = torch.sin(positions * div_term)
        positional_embeddings[:, 0, 1::2] = torch.cos(positions * div_term)
        if self.batch_first:
            positional_embeddings = positional_embeddings.transpose(
                0, 1
            )  # (1, max_len, embed_dim)

        # pe is not learnable state tensor thus let's put it into persistent buffer
        self.register_buffer("positional_embeddings", positional_embeddings)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1) if self.batch_first else x.size(0)
        assert (
            self.max_len >= seq_len
        ), f"max_len smaller than input seq length: {self.max_len} < {seq_len}"

        if self.batch_first:  # x shape: (batch_size, seq_len, embed_dim)
            x = x + self.positional_embeddings[:, :seq_len]
        else:  # x shape: (seq_len, batch_size, embed_dim)
            x = x + self.positional_embeddings[:seq_len]
        return self.dropout(x)


class LookupPositionalEncoding(nn.Module):
    """An absolute positional embedding using a learnable lookup table"""

    def __init__(
        self, embed_dim: int, dropout: float = 0.0, max_len: int = 5000, batch_first: bool = False
    ):
        """
        Args:
            embed_dim (int): embed_dim of input, i.e. d_model for transformer's encoder.
            dropout (float): dropout rate. Default: 0.0.
            max_len (int): maximum sequence length. Default: 5000.
            batch_first (bool): whether the 0th dim of the input is the batch dim. Default: False

        Example usage:
            x = torch.randn(100, 32, 30)
            x_batch_first = x.transpose(0, 1)
            LookupPositionalEncoding(embed_dim=30, max_len=500, batch_first=False)(x)
            LookupPositionalEncoding(embed_dim=30, max_len=500, batch_first=True)(x_batch_first)
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len
        self.positional_embeddings = nn.Embedding(
            self.max_len, embed_dim
        )  # Learnable lookup table for position indices.
        self.batch_first = batch_first

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1) if self.batch_first else x.size(0)
        assert (
            self.max_len >= seq_len
        ), f"max_len smaller than input seq length: {self.max_len} < {seq_len}"

        # Generate absolute position indices
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)  # shape of (1, seq_len)

        # Lookup table for positional embeddings
        positional_embeddings = self.positional_embeddings(
            positions
        )  # shape of (1, seq_len, embed_dim)
        if not self.batch_first:
            positional_embeddings = positional_embeddings.transpose(
                0, 1
            )  # shape of (seq_len, 1, embed_dim)

        x = x + positional_embeddings
        return self.dropout(x)
