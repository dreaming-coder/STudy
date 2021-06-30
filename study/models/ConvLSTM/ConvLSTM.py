from typing import List, Tuple

from ConvLSTMCell import ConvLSTMCell
import torch
from torch import nn, Tensor

__all__ = ["ConvLSTM"]


class ConvLSTM(nn.Module):
    def __init__(self, in_channels: int = 1, hidden_channels_list=None, size: Tuple[int, int] = (100, 100),
                 kernel_size_list=None, forget_bias: float = 0.01):
        super().__init__()
        if hidden_channels_list is None:
            hidden_channels_list = [96, 96]
        if kernel_size_list is None:
            kernel_size_list = [3, 3]

        self.encoder = Encoder(in_channels=in_channels, hidden_channels_list=hidden_channels_list, size=size,
                               kernel_size_list=kernel_size_list, forget_bias=forget_bias)

        self.forecast = Forecast(in_channels=in_channels, hidden_channels_list=hidden_channels_list, size=size,
                                 kernel_size_list=kernel_size_list, forget_bias=forget_bias)

    def forward(self, inputs: Tensor, out_len: int = 10) -> Tensor:
        states = self.encoder(inputs)

        prediction = self.forecast(*states, out_len=out_len)

        return prediction


class Encoder(nn.Module):

    def __init__(self, in_channels: int, hidden_channels_list: List[int], size: Tuple[int, int],
                 kernel_size_list: List[int], forget_bias: float = 0.01):
        """
        :param in_channels:                输入的通道数
        :param hidden_channels_list:       每一层隐藏层的通道数
        :param size:                       输入的尺寸, (Height, Width)
        :param kernel_size_list:           每一层卷积核尺寸
        :param forget_bias:                偏移量
        """
        super(Encoder, self).__init__()

        self.hidden_channels_list = hidden_channels_list
        self.layers = len(hidden_channels_list)

        # 根据堆叠层数，构造ConvLSTMCell列表，加入到模型中
        cell_list = nn.ModuleList([])
        for i in range(self.layers):
            input_channels = in_channels if i == 0 else hidden_channels_list[i - 1]
            cell_list.append(
                ConvLSTMCell(in_channels=input_channels, hidden_channels=hidden_channels_list[i], size=size,
                             kernel_size=kernel_size_list[i], forget_bias=forget_bias)
            )

            self.encoder = cell_list

    def forward(self, inputs: Tensor) -> Tuple[List[Tensor], List[Tensor]]:
        """
        :param inputs: 输入的一个batch的时序数据，shape 为 (B, S, C, H, W)
        :return: 编码阶段之后的 h 和 c
        """
        device = inputs.device

        batch, sequence, channel, height, width = inputs.shape

        # 定义空列表，用于存储每个堆叠层的隐藏状态
        h = []
        c = []
        # 初始化最开始的隐藏状态
        for i in range(self.layers):
            zero_tensor_h = torch.zeros(batch, self.hidden_channels_list[i], height, width).to(device)
            zero_tensor_c = torch.zeros(batch, self.hidden_channels_list[i], height, width).to(device)
            h.append(zero_tensor_h)
            c.append(zero_tensor_c)

        # 沿着时间维度循环
        for s in range(sequence):
            x = inputs[:, s]

            h[0], c[0] = self.encoder[0](x, h[0], c[0])

            for i in range(1, self.layers):
                h[i], c[i] = self.encoder[i](h[i - 1], h[i], c[i])

        return h, c


class Forecast(nn.Module):
    def __init__(self, in_channels: int, hidden_channels_list: List[int], size: Tuple[int, int],
                 kernel_size_list: List[int], forget_bias: float = 0.01):
        r"""
        :param in_channels:              输入通道数
        :param hidden_channels_list:     隐藏层通道数列表
        :param size:                       输入的尺寸, (Height, Width)
        :param kernel_size_list:         卷积核列名
        :param forget_bias:              偏移量
        """
        super(Forecast, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels_list = hidden_channels_list
        self.layers = len(hidden_channels_list)
        self.forget_bias = forget_bias

        # 定义列秒存储堆叠的ConvLSTMCell
        cell_list = nn.ModuleList([])
        for i in range(self.layers):
            input_channels = in_channels if i == 0 else hidden_channels_list[i - 1]
            cell_list.append(
                ConvLSTMCell(in_channels=input_channels, hidden_channels=hidden_channels_list[i], size=size,
                             kernel_size=kernel_size_list[i], forget_bias=forget_bias)
            )

        self.forecast = cell_list

        # 最终输出的通道数要和输入的通道数相同
        self.conv_last = nn.Conv2d(in_channels=sum(hidden_channels_list), out_channels=in_channels,
                                   kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)

    def forward(self, h: List[Tensor], c: List[Tensor], out_len: int = 10) -> Tensor:
        r"""
        :param h:          隐藏层列表
        :param c:          cell列表
        :param out_len:    预测的长度
        :return:           预测的frame
        """
        batch, _, height, width = h[0].shape

        prediction = []

        for _ in range(out_len):
            x = torch.zeros(batch, self.in_channels, height, width).to(h[0].device)

            h[0], c[0] = self.forecast[0](x, h[0], c[0])

            for i in range(1, self.layers):
                h[i], c[i] = self.forecast[i](h[i - 1], h[i], c[i])

            h_concat = torch.cat(h, dim=1)

            pred = self.conv_last(h_concat)

            prediction.append(pred)

        prediction = torch.stack(prediction, dim=0).permute(1, 0, 2, 3, 4)

        return prediction

# if __name__ == '__main__':
#     device = "cuda"
#     net = ConvLSTM().to(device)
#     inputs = torch.ones(7, 10, 1, 100, 100).to(device)
#
#     results = net(inputs, 10)
#
#     print(results.shape)
#     results.sum().backward()
