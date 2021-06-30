from typing import List

import torch
from torch import nn, Tensor

from CausalLSTM import CausalLSTM
from GradientHighwayUnit import GradientHighwayUnit

__all__ = ["PredRNNpp"]


class PredRNNpp(nn.Module):
    def __init__(self, in_channels: int, hidden_channels_list: List[int], ghu_hidden_channels: int,
                 kernel_size_list: List[int], ghu_kernel_size: int, forget_bias: float = 0.01):
        r"""
        :param in_channels:                        输入图片的通道
        :param hidden_channels_list:               每一个堆叠层的隐藏层通道数
        :param ghu_hidden_channels:                GHU的隐藏层通道数
        :param kernel_size_list:                   卷积核尺寸
        :param ghu_kernel_size:                    GHU的卷积核尺寸
        :param forget_bias:                        偏移量
        """
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels_list = hidden_channels_list
        self.ghu_hidden_channels = ghu_hidden_channels
        self.layers = len(hidden_channels_list)
        self.ghu = GradientHighwayUnit(hidden_channels_list[0], ghu_hidden_channels, kernel_size=ghu_kernel_size)
        self.forget_bias = forget_bias

        cell_list = nn.ModuleList([])
        for i in range(self.layers):
            if i == 0:
                cell_list.append(
                    CausalLSTM(in_channels=in_channels, hidden_channels=hidden_channels_list[0],
                               out_channels=hidden_channels_list[1],
                               kernel_size=kernel_size_list[0], forget_bias=forget_bias)
                )
            elif i == 1:
                cell_list.append(
                    CausalLSTM(in_channels=ghu_hidden_channels, hidden_channels=hidden_channels_list[1],
                               out_channels=hidden_channels_list[2],
                               kernel_size=kernel_size_list[1], forget_bias=forget_bias)
                )
            else:
                cell_list.append(
                    CausalLSTM(in_channels=hidden_channels_list[i - 1], hidden_channels=hidden_channels_list[i],
                               out_channels=hidden_channels_list[(i + 1) % self.layers],
                               kernel_size=kernel_size_list[i], forget_bias=forget_bias)
                )

        self.cell_list = cell_list

        self.conv_last = nn.Conv2d(in_channels=hidden_channels_list[-1], out_channels=in_channels,
                                   kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)

    # noinspection PyUnboundLocalVariable
    def forward(self, inputs: Tensor, out_len: int = 10) -> Tensor:
        r"""
        :param inputs:      输入序列
        :param out_len:     预测长度
        :return:            预测的序列
        """
        device = inputs.device
        batch, sequence, channel, height, width = inputs.shape
        h = []
        c = []
        pred = []

        # 初始化最开始的隐藏状态
        for i in range(self.layers):
            zero_tensor_h = torch.zeros(batch, self.hidden_channels_list[i], height, width).to(device)
            zero_tensor_c = torch.zeros(batch, self.hidden_channels_list[i], height, width).to(device)
            h.append(zero_tensor_h)
            c.append(zero_tensor_c)

        m = torch.zeros(batch, self.hidden_channels_list[0], height, width).to(device)
        z = torch.zeros(batch, self.ghu_hidden_channels, height, width).to(device)

        for s in range(sequence + out_len):
            if s < sequence:
                x = inputs[:, s]
            else:
                x = x_pred

            h[0], c[0], m = self.cell_list[0](x, h[0], c[0], m)

            z = self.ghu(h[0], z)

            h[1], c[1], m = self.cell_list[1](z, h[1], c[1], m)

            for i in range(2, self.layers):
                h[i], c[i], m = self.cell_list[i](h[i - 1], h[i], c[i], m)

            x_pred = self.conv_last(h[self.layers - 1])

            if s >= sequence:
                pred.append(x_pred)

        # [length, batch, channel, height, width] -> [batch, length, channel, height, width]
        prediction = torch.stack(pred, dim=0).permute(1, 0, 2, 3, 4)

        return prediction


# if __name__ == '__main__':
#     rnn = PredRNNpp(in_channels=1, hidden_channels_list=[16, 8, 8], ghu_hidden_channels=16,
#                     kernel_size_list=[5, 5, 5], ghu_kernel_size=5).to("cuda")
#     inputs = torch.ones(2, 10, 1, 128, 128).to("cuda")
#
#     result = rnn(inputs, out_len=13)
#     print(result.shape)
#
#     result.sum().backward()
