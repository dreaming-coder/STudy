import torch
from torch import nn, Tensor

__all__ = ["GradientHighwayUnit"]


class GradientHighwayUnit(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int):
        r"""
        :param in_channels:       输入通道
        :param hidden_channels:   状态通道
        :param kernel_size:       卷积核
        """
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels

        padding = (kernel_size // 2, kernel_size // 2)
        kernel_size = (kernel_size, kernel_size)

        self.conv_x = nn.Conv2d(
            in_channels=in_channels, out_channels=hidden_channels * 2,
            kernel_size=kernel_size, padding=padding, stride=(1, 1), bias=False
        )

        self.conv_z = nn.Conv2d(
            in_channels=hidden_channels, out_channels=hidden_channels * 2,
            kernel_size=kernel_size, padding=padding, stride=(1, 1), bias=False
        )

    def forward(self, x: Tensor, z: Tensor) -> Tensor:
        r"""
        :param x:   输入的 Tensor
        :param z:   GHU 的状态 Tensor
        :return:    z
        """
        x_concat = self.conv_x(x)
        x_concat = torch.layer_norm(x_concat, x_concat.shape[1:])
        z_concat = self.conv_z(z)
        z_concat = torch.layer_norm(z_concat, z_concat.shape[1:])

        p_x, s_x = torch.split(x_concat, self.hidden_channels, dim=1)
        p_z, s_z = torch.split(z_concat, self.hidden_channels, dim=1)

        p = torch.tanh(p_x + p_z)
        s = torch.sigmoid(s_x + s_z)

        z = s * p + (1 - s) * z

        return z


# if __name__ == '__main__':
#     ghu = GradientHighwayUnit(in_channels=32, hidden_channels=64, kernel_size=5).to("cuda")
#     x = torch.ones(2, 32, 128, 128).to("cuda")
#     z = torch.ones(2, 64, 128, 128).to("cuda")
#
#     result = ghu(x, z)
#     print(result.shape)
