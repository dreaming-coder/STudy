from typing import List

import torch
from torch import nn, Tensor

from MIMBlock import MIMBlock
from SpatiotemporalLSTM import SpatiotemporalLSTM

__all__ = ["MIM"]


class MIM(nn.Module):

    def __init__(self, in_channels: int, hidden_channels_list: List[int], kernel_size_list: List[int],
                 forget_bias: float = 0.01):
        r"""
        :param in_channels:                  输入通道数
        :param hidden_channels_list:         每一堆叠层的隐藏层通道数
        :param kernel_size_list:             每一堆叠层的卷积核尺寸
        :param forget_bias:                  偏移量
        """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels_list = hidden_channels_list
        self.layers = len(hidden_channels_list)

        cell_list = nn.ModuleList([])

        cell_list.append(
            SpatiotemporalLSTM(
                in_channels=in_channels, hidden_channels=hidden_channels_list[0],
                kernel_size=kernel_size_list[0], forget_bias=forget_bias
            )
        )

        for i in range(1, self.layers):
            cell_list.append(
                MIMBlock(
                    in_channels=hidden_channels_list[i - 1], hidden_channels=hidden_channels_list[i],
                    kernel_size=kernel_size_list[i], forget_bias=forget_bias
                )
            )

        self.cell_list = cell_list

        self.conv_last = nn.Conv2d(
            in_channels=hidden_channels_list[-1], out_channels=in_channels,
            kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False
        )

    # noinspection PyUnboundLocalVariable
    def forward(self, inputs: Tensor, out_len: int = 10) -> Tensor:
        device = inputs.device

        batch, sequence, _, height, width = inputs.shape

        h = []
        h_ = []
        c = []

        # n 和 s 第二层开始才有，故列表第一个元素置为 None
        n = []
        s = []
        pred = []

        # 初始化最开始的隐藏状态
        for i in range(self.layers):
            zero_tensor_h = torch.zeros(batch, self.hidden_channels_list[i], height, width).to(device)
            zero_tensor_c = torch.zeros(batch, self.hidden_channels_list[i], height, width).to(device)
            zero_tensor_h_ = torch.zeros(batch, self.hidden_channels_list[i], height, width).to(device)
            h.append(zero_tensor_h)
            c.append(zero_tensor_c)
            h_.append(zero_tensor_h_)

            if i > 0:
                zero_tensor_n = torch.zeros(batch, self.hidden_channels_list[i], height, width).to(device)
                zero_tensor_s = torch.zeros(batch, self.hidden_channels_list[i], height, width).to(device)
                n.append(zero_tensor_n)
                s.append(zero_tensor_s)
            else:
                n.append(None)
                s.append(None)

        m = torch.zeros(batch, self.hidden_channels_list[0], height, width).to(device)

        for seq in range(sequence + out_len):
            if seq < sequence:
                x = inputs[:, seq]
            else:
                x = x_pred
            h[0], c[0], m = self.cell_list[0](x, h[0], c[0], m)
            h_[0] = h[0]
            for i in range(1, self.layers):
                c[i], h[i], m, n[i], s[i] = self.cell_list[i](h_[i - 1], h[i - 1], c[i], h[i], m, n[i], s[i])
                h_[i] = h[i]

            x_pred = self.conv_last(h[self.layers - 1])

            if seq >= sequence:
                pred.append(x_pred)

        # [length, batch, channel, height, width] -> [batch, length, channel, height, width]
        prediction = torch.stack(pred, dim=0).permute(1, 0, 2, 3, 4)

        return prediction


# if __name__ == '__main__':
#     mim = MIM(in_channels=1, hidden_channels_list=[4, 4, 4, 4], kernel_size_list=[3, 3, 3, 3]).to("cuda")
#     inputs = torch.ones(2, 10, 1, 100, 100).to("cuda")
#
#     result = mim(inputs, out_len=13)
#     print(result.shape)
#
#     result.sum().backward()
