from typing import Tuple
import torch
from torch import nn, Tensor
from TrajGRUCell import TrajGRUCell

__all__ = ["TrajGRU"]


class TrajGRU(nn.Module):

    def __init__(self, in_channels=1):
        super().__init__()
        self.encoder = Encoder(in_channels=in_channels)
        self.forecast = Forecast()

    def forward(self, x: Tensor, out_len: int = 10) -> Tensor:
        r"""
        :param x:
        :param out_len:
        :return:
        """
        assert x.shape[3] == 128 and x.shape[4] == 128, "当前只支持尺寸为128的张量，具体应用请自行计算反卷积的参数"
        x = self.encoder(x)
        x = self.forecast(x, out_len=out_len)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=(7, 7),
                      stride=(3, 3), padding=(1, 1)),  # 输出 41 x 41
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.layer1 = TrajGRUCell(in_channels=8, hidden_channels=64, kernel_size=5, L=13)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=(4, 4),
                      stride=(2, 2), padding=(1, 1)),  # 输出 20 x 20
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.layer2 = TrajGRUCell(in_channels=192, hidden_channels=192, kernel_size=5, L=13)

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(3, 3),
                      stride=(2, 2), padding=(0, 0)),  # 输出 9 x 9
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.layer3 = TrajGRUCell(in_channels=192, hidden_channels=192, kernel_size=3, L=9)

    def forward(self, input_seq: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # 将input转换为[S, B, C, H, W]
        input_seq = input_seq.transpose(1, 0)

        h_1 = None
        h_2 = None
        h_3 = None
        for seq in input_seq:
            seq = self.conv1(seq)
            seq, h_1 = self.layer1(seq, h_1)
            seq = self.conv2(seq)
            seq, h_2 = self.layer2(seq, h_2)
            seq = self.conv3(seq)
            _, h_3 = self.layer3(seq, h_3)

        return h_3, h_2, h_1


class Forecast(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = TrajGRUCell(in_channels=192, hidden_channels=192, kernel_size=5, L=13)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=192, out_channels=192, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0)),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.layer2 = TrajGRUCell(in_channels=192, hidden_channels=192, kernel_size=5, L=13)
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=192, out_channels=64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.layer3 = TrajGRUCell(in_channels=64, hidden_channels=64, kernel_size=3, L=9)
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=8, kernel_size=(7, 7), stride=(3, 3), padding=(1, 1)),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        )

    def forward(self, h_states: Tensor, out_len: int = 10) -> Tensor:
        r"""
        :param h_states:
        :param out_len:
        :return:
        """
        h_1, h_2, h_3 = h_states
        results = []
        ret = None
        for _ in range(out_len):
            ret, h_1 = self.layer1(ret, h_1)

            ret = self.deconv1(ret)

            ret, h_2 = self.layer2(ret, h_2)

            ret = self.deconv2(ret)
            ret, h_3 = self.layer3(ret, h_3)
            ret = self.deconv3(ret)
            results.append(ret.unsqueeze(0))
            ret = None

        results = torch.cat(results)
        results = results.transpose(1, 0)  # 转成 [B, S, C, H, W]

        return results


# if __name__ == '__main__':
#     net = TrajGRU(in_channels=1).cuda()
#     x = torch.ones(2, 10, 1, 128, 128).cuda()
#     result = net(x, out_len=20)
#     print(result.shape)
#
#     result.sum().backward()
