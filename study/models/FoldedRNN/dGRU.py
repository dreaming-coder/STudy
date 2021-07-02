from typing import List

import torch
from torch import nn


class dGRU(nn.Module):

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int):
        super(dGRU, self).__init__()

        self.state_lower = None
        self.hidden_channels = hidden_channels

        self.state_upper = None

        padding = (kernel_size // 2, kernel_size // 2)
        kernel_size = (kernel_size, kernel_size)

        self.conv_r = nn.Conv2d(
            in_channels=in_channels + hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size,
            stride=(1, 1), padding=padding
        )

        self.conv_z = nn.Conv2d(
            in_channels=in_channels + hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size,
            stride=(1, 1), padding=padding
        )

        self.conv_h = nn.Conv2d(
            in_channels=in_channels + hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size,
            stride=(1, 1), padding=padding
        )

        self.conv_r_back = nn.Conv2d(
            in_channels=in_channels + hidden_channels, out_channels=in_channels, kernel_size=kernel_size,
            stride=(1, 1), padding=padding
        )

        self.conv_z_back = nn.Conv2d(
            in_channels=in_channels + hidden_channels, out_channels=in_channels, kernel_size=kernel_size,
            stride=(1, 1), padding=padding
        )

        self.conv_h_back = nn.Conv2d(
            in_channels=in_channels + hidden_channels, out_channels=in_channels, kernel_size=kernel_size,
            stride=(1, 1), padding=padding
        )

    def forward(self, x):
        self.state_lower = x
        if self.state_upper is None:
            self.state_upper = torch.zeros(x.shape[0], self.hidden_channels, x.shape[2], x.shape[3]).to(x.device)

        x1 = [x, self.state_upper]
        x1 = torch.cat(tuple(x1), dim=1)

        r = torch.sigmoid(self.conv_r(x1))
        z = torch.sigmoid(self.conv_z(x1))

        x2 = torch.cat([x, self.state_upper * r], dim=1)
        hh = torch.tanh(self.conv_h(x2))

        self.state_upper = z * hh + (1 - z) * self.state_upper

        return self.state_upper

    def backward(self, x):
        x1 = [self.state_lower, x]
        x1 = torch.cat(tuple(x1), dim=1)

        r = torch.sigmoid(self.conv_r_back(x1))
        z = torch.sigmoid(self.conv_z_back(x1))

        x2 = torch.cat([self.state_lower * r, x], dim=1)
        hh = torch.tanh(self.conv_h_back(x2))

        self.state_lower = z * hh + (1 - z) * self.state_lower

        return self.state_lower


if __name__ == '__main__':
    gru = dGRU(in_channels=1, hidden_channels=16, kernel_size=5).cuda()
    f = torch.ones(3, 1, 100, 100).cuda()
    b = torch.ones(3, 16, 100, 100).cuda()

    r = gru.forward(f)
    print(r.shape)
    r2 = gru.backward(b)
    print(r2.shape)
