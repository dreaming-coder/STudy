from collections import deque

from CubicLSTM import CubicLSTM

import torch
from torch import nn, Tensor

__all__ = ["CubicRNN"]


class CubicRNN(nn.Module):

    def __init__(self, in_channels: int, hidden_channels: int,
                 spatial_layers: int = 3, output_layers: int = 3,
                 kernel_size_x: int = 3, kernel_size_y: int = 1, kernel_size_z: int = 5,
                 forget_bias: float = 0.01):
        r"""
        :param in_channels:
        :param hidden_channels:
        :param spatial_layers:
        :param output_layers:
        :param kernel_size_x:
        :param kernel_size_y:
        :param kernel_size_z:
        :param forget_bias:
        """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.output_layers = output_layers
        self.spatial_layers = spatial_layers

        cell_list = nn.ModuleList([])
        for i in range(output_layers):
            cell_list.append(nn.ModuleList([]))
            for j in range(spatial_layers):
                if i == 0:
                    cell_list[i].append(
                        CubicLSTM(
                            in_channels=in_channels, hidden_channels=hidden_channels, kernel_size_x=kernel_size_x,
                            kernel_size_y=kernel_size_y, kernel_size_z=kernel_size_z, forget_bias=forget_bias
                        )
                    )
                else:
                    cell_list[i].append(
                        CubicLSTM(
                            in_channels=hidden_channels, hidden_channels=hidden_channels, kernel_size_x=kernel_size_x,
                            kernel_size_y=kernel_size_y, kernel_size_z=kernel_size_z, forget_bias=forget_bias
                        )
                    )
        self.cell_list = cell_list

        self.conv_last = nn.Conv2d(in_channels=hidden_channels, out_channels=in_channels, kernel_size=(1, 1),
                                   stride=(1, 1), padding=(0, 0), bias=False)

    # noinspection PyUnboundLocalVariable,PyTypeChecker
    def forward(self, input_seq: Tensor, out_len: int = 10) -> Tensor:
        batch, sequence, channel, height, width = input_seq.shape
        device = input_seq.device

        state_z_list = []
        state_x_list = []
        y_list = [None] * self.spatial_layers
        prediction = []

        # 处理空间维度的分支
        for i in range(self.output_layers):
            state_z = torch.zeros(batch, self.hidden_channels * 2, height, width).to(device)
            state_z_list.append(state_z)

        # 处理时间维度的分支
        for i in range(self.output_layers):
            state_x_list.append([])
            for j in range(self.spatial_layers):
                state_x = torch.zeros(batch, self.hidden_channels * 2, height, width).to(device)
                state_x_list[i].append(state_x)

        # 定义窗口队列，固定长度，并初始化 【最大窗口长度-1】，进一个出一个
        window = deque([input_seq[:, i, ] for i in range(self.spatial_layers - 1)], maxlen=self.spatial_layers)
        # 这层 for 循环控制时序次数
        for index in range(self.spatial_layers - 1, sequence + out_len):
            if index < sequence:
                candidate = input_seq[:, index]
            else:
                candidate = pred

            window.append(candidate)
            for l in range(self.output_layers):
                if l == 0:
                    inputs = window
                else:
                    inputs = y_list

                y_list[0], state_x_list[l][0], state_z_list[l] = self.cell_list[l][0](
                    inputs[0], state_x_list[l][0], state_z_list[l]
                )
                for s in range(1, self.spatial_layers):
                    y_list[s], state_x_list[l][s], state_z_list[l] = self.cell_list[l][s](
                        inputs[s], state_x_list[l][s - 1], state_z_list[l]
                    )

            pred = self.conv_last(y_list[-1])

            if index >= sequence:
                prediction.append(pred)

        # [length, batch, channel, height, width] -> [batch, length, channel, height, width]
        prediction = torch.stack(prediction, dim=0).permute(1, 0, 2, 3, 4)

        return prediction


# if __name__ == '__main__':
#     rnn = CubicRNN(in_channels=4, hidden_channels=32, spatial_layers=3, output_layers=2,
#                    kernel_size_x=3, kernel_size_y=1, kernel_size_z=5, forget_bias=0.1).to("cuda")
#     x = torch.randint(0, 60, (2, 10, 4, 128, 128)).float().to("cuda")
#     r = rnn(x, out_len=12)
#     print(r.shape)
#
#     r.sum().backward()
