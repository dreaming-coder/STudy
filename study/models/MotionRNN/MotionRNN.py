from typing import List

from MotionGRU import MotionGRU
from SpatiotemporalLSTM import SpatiotemporalLSTM
import torch
from torch import nn, Tensor


# 由于 M 的存在之字形贯穿所有 cell，要保证 M 的 channel 不变，只能所有 layer 隐藏层通道数相同
class MotionRNN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels_list: List[int], kernel_size_list: List[int],
                 forget_bias: float = 0.01, k: int = 3, alpha=0.5):
        r"""
        :param in_channels:               输入帧的通道数
        :param hidden_channels_list:      每一个堆叠层的隐藏层通道数
        :param kernel_size_list:          每一个堆叠层的卷积核尺寸
        :param forget_bias:               偏移量
        """
        super(MotionRNN, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels_list = hidden_channels_list
        self.layers = len(hidden_channels_list)
        self.forget_bias = forget_bias
        self.k = k

        cell_list = nn.ModuleList([])
        motion_gru = nn.ModuleList([])
        for i in range(self.layers):
            input_channels = in_channels if i == 0 else hidden_channels_list[i - 1]
            cell_list.append(
                SpatiotemporalLSTM(in_channels=input_channels, hidden_channels=hidden_channels_list[i],
                                   kernel_size=kernel_size_list[i], forget_bias=forget_bias)
            )
            if i < self.layers - 1:
                motion_gru.append(
                    MotionGRU(hidden_channels=hidden_channels_list[i], k=k, alpha=alpha)
                )

        self.cell_list = cell_list
        self.motion_gru = motion_gru

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
        o = []  # 输出门
        d = [None]
        f = [None]
        pred = []  # 存储预测结果

        # 初始化最开始的隐藏状态
        for i in range(self.layers):
            zero_tensor_h = torch.zeros(batch, self.hidden_channels_list[i], height, width).to(device)
            zero_tensor_c = torch.zeros(batch, self.hidden_channels_list[i], height, width).to(device)
            zero_tensor_o = torch.zeros(batch, self.hidden_channels_list[i], height, width).to(device)
            h.append(zero_tensor_h)
            c.append(zero_tensor_c)
            o.append(zero_tensor_o)

        for j in range(1, self.layers):
            zero_tensor_d = torch.zeros(batch, 2 * self.k ** 2, height // 2, width // 2).to(device)
            zero_tensor_f = torch.zeros(batch, 2 * self.k ** 2, height // 2, width // 2).to(device)
            d.append(zero_tensor_d)
            f.append(zero_tensor_f)

        m = torch.zeros(batch, self.hidden_channels_list[0], height, width).to(device)

        # 开始循环，模型在预测部分的输入是前一帧的预测输出
        for s in range(sequence + out_len):
            if s < sequence:
                x = inputs[:, s]
            else:
                x = x_pred

            h[0], c[0], m, o[0] = self.cell_list[0](x, h[0], c[0], m)

            for i in range(1, self.layers):
                xx, f[i], d[i] = self.motion_gru[i - 1](h[i - 1], f[i], d[i])
                h[i], c[i], m, o[0] = self.cell_list[i](xx, h[i], c[i], m)
                h[i] = h[i] + (1 - o[i]) * h[i - 1]

            x_pred = self.conv_last(h[self.layers - 1])

            if s >= sequence:
                pred.append(x_pred)

        # [length, batch, channel, height, width] -> [batch, length, channel, height, width]
        prediction = torch.stack(pred, dim=0).permute(1, 0, 2, 3, 4)

        return prediction


if __name__ == '__main__':
    net = MotionRNN(in_channels=1, hidden_channels_list=[32, 32, 32, 32], kernel_size_list=[3, 3, 3, 3]).to("cuda")
    inputs = torch.ones(2, 10, 1, 100, 100).to("cuda")
    result = net(inputs, out_len=12)
    print(result.shape)
    result.sum().backward()
