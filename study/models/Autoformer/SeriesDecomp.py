import math

import torch
from torch import nn, Tensor
import torch.nn.functional as F

__all__ = ["SeriesDecomp"]


class SeriesDecomp(nn.Module):
    def __init__(self):
        super(SeriesDecomp, self).__init__()

    def forward(self, x: Tensor):
        """
        :param x:  (batch, d, L)
        :return: x_s, x_t  (batch, d ,L)
        """
        batch, dimension, length = x.shape
        padding = math.ceil((length - 1) / 2)
        scalar_pre = x[..., 0].reshape(batch, dimension, 1).repeat(1, 1, padding)
        scalar_suf = x[..., -1].reshape(batch, dimension, 1).repeat(1, 1, padding)
        padding_x = torch.cat([scalar_pre, x, scalar_suf], dim=-1)
        x_t = F.avg_pool1d(padding_x, padding * 2 + 1, stride=1)
        x_s = x - x_t
        return x_s, x_t


# if __name__ == '__main__':
#     d = 7
#     l = 25
#     a = torch.rand(size=(2, d, l))
#     print(a.shape)
#     dec = SeriesDecomp()
#     c = dec(a)
#     print(c[0].shape)
#     print(c[1].shape)
