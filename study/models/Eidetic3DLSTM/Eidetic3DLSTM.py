from collections import deque
from typing import List, Tuple

from Eidetic3DLSTMCell import Eidetic3DLSTMCell
import torch
from torch import nn, Tensor

__all__ = ["Eidetic3DLSTM"]


# noinspection PyTypeChecker
class Eidetic3DLSTM(nn.Module):

    def __init__(self, in_channels: int, hidden_channels_list: List[int], window_length: int, kernel_size: Tuple):
        r"""
        :param in_channels:
        :param hidden_channels_list:
        :param window_length:
        :param kernel_size:
        """
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels_list = hidden_channels_list
        self.layers = len(hidden_channels_list)
        self.window_length = window_length

        cell_list = nn.ModuleList([])
        for i in range(self.layers):
            input_channel = in_channels if i == 0 else self.hidden_channels_list[i - 1]
            cell_list.append(
                Eidetic3DLSTMCell(in_channels=input_channel, hidden_channels=self.hidden_channels_list[i],
                                  depth=self.window_length, kernel_size=kernel_size)
            )
        self.cell_list = cell_list

        self.conv_last = nn.Conv3d(in_channels=self.hidden_channels_list[-1], out_channels=in_channels,
                                   kernel_size=(window_length, 1, 1), stride=1, padding=0)

    # noinspection PyUnboundLocalVariable
    def forward(self, input_seq: Tensor, out_len: int = 10) -> Tensor:
        r"""
        :param input_seq:
        :param out_len:
        :return:
        """
        device = input_seq.device
        batch, sequence, _, height, width = input_seq.shape

        c_states, h, pred = [], [], []

        # 初始化最开始的隐藏状态
        for i in range(self.layers):
            zero_state_h = torch.zeros(batch, self.hidden_channels_list[i],
                                       self.window_length, height, width).to(device)
            zero_state_c = torch.zeros(batch, self.hidden_channels_list[i],
                                       self.window_length, height, width).to(device)

            c_states.append([zero_state_c])
            h.append(zero_state_h)

        m = torch.zeros(batch, self.hidden_channels_list[0], self.window_length, height, width).to(device)

        input_queue = deque(maxlen=self.window_length)

        for time_step in range(self.window_length - 1):
            input_queue.append(
                torch.zeros(batch, self.in_channels, height, width).to(device)
            )

        for time_step in range(sequence + out_len):
            if time_step < sequence:
                x = input_seq[:, time_step]
            else:
                x = x_pred

            input_queue.append(x)

            x = torch.stack(tuple(input_queue))
            x = x.permute(1, 2, 0, 3, 4)

            for i in range(self.layers):
                if i == 0:
                    inputs = x
                else:
                    inputs = h[i - 1]

                c, h[i], m = self.cell_list[i](inputs, h[i], c_states[i], m)
                c_states[i].append(c)

            x_pred = self.conv_last(h[-1]).squeeze(dim=2)  # [batch, channel, height, width]
            pred.append(x_pred)

        # [length, batch, channel, height, width] -> [batch, length, channel, height, width]
        prediction = torch.stack(pred[-out_len:], dim=0).permute(1, 0, 2, 3, 4)

        return prediction


# if __name__ == '__main__':
#     net = Eidetic3DLSTM(in_channels=1, hidden_channels_list=[4],
#                         window_length=2, kernel_size=(2, 5, 5)).to("cuda")
#
#     inputs = torch.ones(1, 10, 1, 25, 25).to("cuda")
#     result = net(inputs, out_len=10)
#     print(result.shape)
