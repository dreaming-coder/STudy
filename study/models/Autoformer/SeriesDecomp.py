import torch
from torch import nn, Tensor

__all__ = ["SeriesDecomp"]


class SeriesDecomp(nn.Module):
    def __init__(self, window: int):
        super(SeriesDecomp, self).__init__()
        self.window = window
        self.avg_pool = nn.AvgPool1d(kernel_size=window, stride=1)

    def forward(self, x: Tensor):
        """
        :param x:  (batch, d, L)
        :return: x_s, x_t  (batch, d ,L)
        """
        batch, dimension, _ = x.shape
        zeros_prefix = torch.zeros(batch, dimension, self.window - 1)
        padding_x = torch.cat([zeros_prefix, x], dim=-1)
        x_t = self.avg_pool(padding_x)
        x_s = x - x_t
        return x_s, x_t


# if __name__ == '__main__':
#     batch, d, l = 3, 5, 20
#     x = torch.ones(batch, d, l)
#     sd = SeriesDecomp(window=5)
#     x_s, x_t = sd(x)
#     print(x_s.shape)  # AvgPool1d
#     print(x_t.shape)  # AvgPool1d
