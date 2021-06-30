from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F

__all__ = ["TrajGRUCell"]


def _warp(input: Tensor, flow: Tensor) -> Tensor:
    """
    这个操作和可变形卷积类似
    """
    device = input.device
    B, C, H, W = input.size()
    # mesh grid

    # 这两行代码很像 broadcasting
    # shape (H, W)
    # 每一行都是 0, 1, 2, ..., W
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1).to(device)

    # shape (H, W)
    # 每一列都是 0, 1, 2, ..., H
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W).to(device)

    # 扩展为 4 维，shape (B, 1, H, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)

    # shape (B, 2, H, W)
    grid = torch.cat((xx, yy), 1).float()

    # 这里注意 grid 和 flow 要能够 broadcasting， 不过讲道理，flow 要求传进来就是这个维度
    vgrid = grid + flow

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    # examples
    # >>> x = torch.randn(2, 3, 5)
    # >>> x.size()
    # torch.Size([2, 3, 5])
    # >>> x.permute(2, 0, 1).size()
    # torch.Size([5, 2, 3])
    vgrid = vgrid.permute(0, 2, 3, 1)  # 处理后 shape 为 (B, H, W, 2)

    # https://blog.csdn.net/chamber_of_secrets/article/details/83512540
    # 对于 output 上的每一个点，（x, y）三个通道的像素值，采集自 input 上某一点三个通道的像素值，
    # 采集哪个点呢，坐标存储在grid最低维，也就是(B x H x W x 2) 中的2，
    # [0] 索引到input的x坐标，[1]索引到input的y坐标
    output = F.grid_sample(input, vgrid, align_corners=True)

    return output


class TrajGRUCell(nn.Module):

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 5, L: int = 5):
        """具体看论文吧"""
        super().__init__()

        assert kernel_size == 3 or kernel_size == 5

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels

        kernel_size = (kernel_size, kernel_size)
        same_padding = tuple([size // 2 for size in kernel_size])
        self.l = L

        # x to flow
        self.gammma_x = nn.Conv2d(in_channels, out_channels=32, kernel_size=(5, 5), padding=(2, 2), stride=(1, 1))

        # h to flow
        self.gammma_h = nn.Conv2d(hidden_channels, out_channels=32, kernel_size=(5, 5), padding=(2, 2), stride=(1, 1))

        # generate flow
        self.generate_flow = nn.Conv2d(in_channels=32, out_channels=self.l * 2, kernel_size=(5, 5), padding=(2, 2),
                                       stride=(1, 1))

        self.W_xz = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels,
                              kernel_size=kernel_size, padding=same_padding, stride=(1, 1))

        self.W_xr = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels,
                              kernel_size=kernel_size, padding=same_padding, stride=(1, 1))

        self.W_xh = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels,
                              kernel_size=kernel_size, padding=same_padding, stride=(1, 1))

        self.W_hz = nn.ModuleList([])
        self.W_hr = nn.ModuleList([])
        self.W_hh = nn.ModuleList([])
        for i in range(self.l):
            self.W_hz.append(nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels,
                                       kernel_size=(1, 1), padding=(0, 0), stride=(1, 1))
                             )

            self.W_hr.append(nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels,
                                       kernel_size=(1, 1), padding=(0, 0), stride=(1, 1))
                             )

            self.W_hh.append(nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels,
                                       kernel_size=(1, 1), padding=(0, 0), stride=(1, 1))
                             )

    # @title                       __flow_generator
    # @description                 论文中的 subnet gamma
    # @author                      ice 2020/04/29 10:35
    # @param
    #     x          Tensor        [B, C=in_channels, H, W]
    #     h          Tensor        [B, C=hidden_channels, H, W]
    # @return
    #     flows      Tuple         每个元素 shape 为 [B, 2, H, W]，共有 self.l 个
    def __flow_generator(self, x: Tensor, h: Tensor) -> Tensor:
        r"""
        :param x:
        :param h:
        :return:
        """
        if x is not None:
            x2f = self.gammma_x(x)
        else:
            x2f = None
        h2f = self.gammma_h(h)
        f = x2f + h2f if x2f is not None else h2f
        f = torch.nn.functional.leaky_relu(f, negative_slope=0.2)

        flows = self.generate_flow(f)
        flows = torch.split(flows, 2, dim=1)

        return flows

        # x 和 h 不同时为空
        # x: [B, C, H, W]
        # h: [B, C, H, W]

    def forward(self, x: Tensor = None, h: Tensor = None) -> Tuple[Tensor, Tensor]:
        r"""
        :param x:
        :param h:
        :return:
        """
        # x 是输入的一个 batch 的某一时序，shape应该是 (B, C, H, W)
        # h 和 c 也设置成这种 shape，因为 PyTorch 好像默认是多传一维 batch_size 的，这样就输入的维度含义一致了
        if x is None and h is None:
            raise ValueError(f"输入 {x} 和隐藏状态 {h} 不能同时为空")

        if x is None:
            x = torch.zeros(h.shape[0], self.in_channels, h.shape[2], h.shape[3]).to(h.device)

        if h is None:
            h = torch.zeros(x.shape[0], self.hidden_channels, x.shape[2], x.shape[3]).to(x.device)

        flows = self.__flow_generator(x, h)
        warped_data = []
        for flow in flows:
            warped_data.append(_warp(h, flow))
        temp_z_t = torch.tensor(0, device=x.device)
        temp_r_t = torch.tensor(0, device=x.device)
        temp_h_t = torch.tensor(0, device=x.device)
        for i in range(self.l):
            temp_z_t = temp_z_t + self.W_hz[i](warped_data[i])
            temp_r_t = temp_r_t + self.W_hr[i](warped_data[i])
            temp_h_t = temp_h_t + self.W_hh[i](warped_data[i])

        z_t = torch.sigmoid(self.W_xz(x) + temp_z_t)
        r_t = torch.sigmoid(self.W_xr(x) + temp_r_t)

        temp = r_t * temp_h_t
        h_temp = torch.nn.functional.leaky_relu(self.W_xh(x) + temp, negative_slope=0.2)

        h = (1 - z_t) * h_temp + z_t * h_temp

        return h, h


# if __name__ == '__main__':
#     cell = TrajGRUCell(in_channels=1, hidden_channels=64, kernel_size=3).cuda()
#     x = torch.ones(2, 1, 128, 128).cuda()
#     h = torch.ones(2, 64, 128, 128).cuda()
#
#     result = cell(x, h)
#     print(result[0].shape)
#     print(result[1].shape)
