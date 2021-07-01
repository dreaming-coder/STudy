from typing import List, Tuple

from torch import nn, Tensor

from SpatiotemporalLSTM import SpatiotemporalLSTM

__all__ = ["ReversiblePredictiveModule"]


class ReversiblePredictiveModule(nn.Module):

    def __init__(self, channels: int, n_layers: int):
        super(ReversiblePredictiveModule, self).__init__()
        self.channels = channels
        self.n_layers = n_layers

        self.ConvRNN = nn.ModuleList([])
        self.attention = nn.ModuleList([])

        for i in range(n_layers):
            self.ConvRNN.append(
                SpatiotemporalLSTM(in_channels=channels, hidden_channels=channels, kernel_size=3)
            )

            self.attention.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels, 1, 1, 0),
                    nn.GroupNorm(1, channels, affine=True),
                    nn.ReLU(),
                    nn.Conv2d(channels, channels, 3, 1, 1),
                    nn.GroupNorm(1, channels, affine=True),
                    nn.ReLU(),
                    nn.Conv2d(channels, channels, 1, 1, 0),
                    nn.Sigmoid()
                )
            )

    def forward(self, inputs: Tuple[Tensor, Tensor], h: List[Tensor], c: List[Tensor], m: Tensor):
        x1, x2 = inputs
        for i in range(self.n_layers):
            h[i], c[i], m = self.ConvRNN[i](x1, h[i], c[i], m)
            g = self.attention[i](h[i])
            x2 = (1 - g) * x2 + g * h[i]
            x1, x2 = x2, x1

        return (x1, x2), h, c, m
