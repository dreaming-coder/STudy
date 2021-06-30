from typing import List

from SpatiotemporalLSTM import SpatiotemporalLSTM
from torch import nn, Tensor
import torch

__all__ = ["PredRNN"]


class PredRNN(nn.Module):

    def __init__(self, in_channels: int, hidden_channels_list: List[int], kernel_size_list: List[int],
                 forget_bias: float = 0.01):
        r"""
        :param in_channels:               输入帧的通道数
        :param hidden_channels_list:      每一个堆叠层的隐藏层通道数
        :param kernel_size_list:          每一个堆叠层的卷积核尺寸
        :param forget_bias:               偏移量
        """
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels_list = hidden_channels_list
        self.layers = len(hidden_channels_list)
        self.forget_bias = forget_bias

        cell_list = nn.ModuleList([])
        for i in range(self.layers):
            input_channels = in_channels if i == 0 else hidden_channels_list[i - 1]
            cell_list.append(
                SpatiotemporalLSTM(in_channels=input_channels, hidden_channels=hidden_channels_list[i],
                                   kernel_size=kernel_size_list[i], forget_bias=forget_bias)
            )

        self.cell_list = cell_list

        # 最后输出的通道数和输入通道数一样
        self.conv_last = nn.Conv2d(in_channels=hidden_channels_list[-1], out_channels=in_channels,
                                   kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=False)

    # noinspection PyUnboundLocalVariable
    def forward(self, inputs: Tensor, out_len: int = 10) -> Tensor:
        r"""
        :param inputs:   输入序列
        :param out_len:  预测长度
        :return:         输出序列
        """
        device = inputs.device
        batch, sequence, channel, height, width = inputs.shape

        h = []  # 存储隐藏层
        c = []  # 存储cell记忆
        pred = []  # 存储预测结果

        # 初始化最开始的隐藏状态
        for i in range(self.layers):
            zero_tensor_h = torch.zeros(batch, self.hidden_channels_list[i], height, width).to(device)
            zero_tensor_c = torch.zeros(batch, self.hidden_channels_list[i], height, width).to(device)
            h.append(zero_tensor_h)
            c.append(zero_tensor_c)

        m = torch.zeros(batch, self.hidden_channels_list[0], height, width).to(device)

        # 开始循环，模型在预测部分的输入是前一帧的预测输出
        for s in range(sequence + out_len):
            if s < sequence:
                x = inputs[:, s]
            else:
                x = x_pred

            h[0], c[0], m = self.cell_list[0](x, h[0], c[0], m)

            for i in range(1, self.layers):
                h[i], c[i], m = self.cell_list[i](h[i - 1], h[i], c[i], m)

            x_pred = self.conv_last(h[self.layers - 1])

            if s >= sequence:
                pred.append(x_pred)

        # [length, batch, channel, height, width] -> [batch, length, channel, height, width]
        prediction = torch.stack(pred, dim=0).permute(1, 0, 2, 3, 4)

        return prediction


# if __name__ == '__main__':
#     net = PredRNN(in_channels=1, hidden_channels_list=[16, 16, 16, 16], kernel_size_list=[3, 3, 3, 3]).to("cuda")
#     inputs = torch.ones(2, 10, 1, 100, 100).to("cuda")
#     result = net(inputs, out_len=12)
#     print(result.shape)
#     result.sum().backward()
