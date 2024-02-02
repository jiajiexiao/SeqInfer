from torch import nn


class BaseEncoder(nn.Module):
    @abstractmethod
    def encoder(self) -> nn.Module:
        raise NotImplementedError()
