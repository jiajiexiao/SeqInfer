import torch
import torch.nn as nn

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
                corresponding module will be imported; when it's None, `torch.nn.Identity`
                will be used. Default to `torch.nn.ReLU`.
            squeeze_output (bool, optional):
                Whether to squeeze the last dimension of the output tensor. Having this option is to
                squeeze the last dim if that dim of x is 1. This operation can be useful when the
                criterion/loss functions require the input and target have the same shape. For
                example, the target can have a shape of (N, ) while the unsqueezed output from this
                FNN is (N, 1). This will not be accepted by loss function like
                `torch.nn.BCEWithLogitsLoss` and squeeze (N, 1) to (N, ) is needed. Defaults to False.
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
            activation = import_object_from_path("torch.nn.Identity")
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
