from typing import Tuple

import torch
from torch import nn, Tensor
from torch.nn import functional as F

__all__ = ["MotionGRU"]


class MotionGRU(nn.Module):

    def __init__(self, hidden_channels, k: int = 3, alpha=0.5):
        super(MotionGRU, self).__init__()
        k = 3 if k != 3 else 3  # 这里写死卷积核就是 3，一来这个用的多，而来这个要是改变了，相应的 stride 和 padding 也会变化

        self.encoder = Encoder(hidden_channels=hidden_channels, k=k)
        self.transient = Transient(hidden_channels=hidden_channels, k=k)
        self.trend = Trend(alpha=alpha)
        self.broadcast = Broadcast(hidden_channels=hidden_channels, k=k)
        self.warp = Warp(k=k)
        self.decoder = Decoder(hidden_channels=hidden_channels, k=k)
        self.conv_1x1 = nn.Conv2d(
            in_channels=hidden_channels * 2, out_channels=hidden_channels,
            kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)
        )

    def forward(self, h, f, d) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            h: h denotes the :math:`H^l_t`
            f: f denotes the :math:`F^l_{t-1}`
            d: d denotes the :math:`D^l_{t-1}`
        """
        enc_h = self.encoder(h)
        f_ = self.transient(f, enc_h)
        d = self.trend(f, d)
        f = f_ + d

        m = self.broadcast(enc_h)
        warped = self.warp(enc_h, f)
        h_ = m * warped
        dec_h = self.decoder(h_)

        g = torch.sigmoid(self.conv_1x1(torch.cat([dec_h, h], dim=1)))

        x = g * h + (1 - g) * dec_h

        return x, f, d


class Encoder(nn.Module):
    def __init__(self, hidden_channels, k: int = 3):
        """
        Args:
            hidden_channels: the number of the channels for the hidden states
            k: the learned filter size of MotionGRU, 3 is the default
        """
        super(Encoder, self).__init__()
        self.encoder = nn.Conv2d(
            in_channels=hidden_channels, out_channels=hidden_channels // 4,
            kernel_size=(k, k), stride=(2, 2), padding=(1, 1)
        )

    def forward(self, h: Tensor) -> Tensor:
        enc_h = self.encoder(h)
        return enc_h


class Transient(nn.Module):
    def __init__(self, hidden_channels, k: int = 3):
        """
        Args:
            hidden_channels: the number of the channels for the hidden states
            k: the learned filter size of MotionGRU, 3 is the default
        """
        super(Transient, self).__init__()

        self.conv_u = nn.Conv2d(
            in_channels=hidden_channels // 4 + 2 * k ** 2, out_channels=2 * k ** 2,
            kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)
        )

        self.conv_r = nn.Conv2d(
            in_channels=hidden_channels // 4 + 2 * k ** 2, out_channels=2 * k ** 2,
            kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)
        )

        self.conv_z = nn.Conv2d(
            in_channels=hidden_channels // 4 + 2 * k ** 2, out_channels=2 * k ** 2,
            kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)
        )

    def forward(self, f, enc_h) -> Tensor:
        """
        Args:
            f: f denotes the :math:`F^l_{t-1}`
            enc_h: h denotes the :math:`Enc(H^l_t)`
        Returns:
            f_: f_ is the :math:`F^'_t`
        """
        u = torch.sigmoid(
            self.conv_u(torch.cat([enc_h, f], dim=1))
        )

        r = torch.sigmoid(
            self.conv_r(torch.cat([enc_h, f], dim=1))
        )

        z = torch.tanh(
            self.conv_z(torch.cat([enc_h, r * f], dim=1))
        )

        f_ = u * z + (1 - u) * f

        return f_


class Trend(nn.Module):
    def __init__(self, alpha=0.5):
        """
        Args:
            alpha: the alpha is the step size of momentum update, default is 0.5
        """
        super(Trend, self).__init__()
        self.alpha = alpha

    def forward(self, f, d) -> Tensor:
        """
        Args:
            f: f denotes the :math:`F^l_{t-1}`
            d: d denotes the :math:`D^l_{t-1}`
        Returns:
            d_: d_ is the :math:`D^l_t`
        """
        d_ = d + self.alpha * (f - d)
        return d_


class Broadcast(nn.Module):
    def __init__(self, hidden_channels, k: int = 3):
        super(Broadcast, self).__init__()
        self.hidden_channels = hidden_channels
        self.conv_m = nn.Conv2d(
            in_channels=hidden_channels // 4, out_channels=k ** 2,
            kernel_size=(k, k), stride=(1, 1), padding=(1, 1)
        )

    def forward(self, enc_h: Tensor) -> Tensor:
        """
        Args:
            enc_h: enc_h denotes the :math:`Enc(H^l_t)`
        Returns:
            m: m is the :math:`m_t`
        """
        m = torch.sigmoid(self.conv_m(enc_h)) \
            .permute(0, 2, 3, 1).unsqueeze(dim=1) \
            .repeat(1, self.hidden_channels // 4, 1, 1, 1)

        return m


class Warp(nn.Module):
    def __init__(self, k: int = 3):
        super(Warp, self).__init__()
        self.k = k

    def forward(self, enc_h: Tensor, f: Tensor) -> Tensor:
        device = enc_h.device
        B, C, H, W = enc_h.shape

        # shape (H, W)
        # 每一行都是 0, 1, 2, ..., W
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1).to(device)

        # shape (H, W)
        # 每一列都是 0, 1, 2, ..., H
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W).to(device)

        # 扩展为 4 维，shape (B, 1, H, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)

        warped_data = []
        for i in range(self.k ** 2):
            # calculate the initial offset,
            p_ix = int(i / self.k) - int(self.k / 2)
            p_iy = (i % self.k) - int(self.k / 2)

            xxx = xx + p_ix
            yyy = yy + p_iy

            # shape (B, 2, H, W)
            grid = torch.cat((xxx, yyy), 1).float()

            # shape (B, 2, H, W)
            flow = f[::, i::self.k ** 2, ...]

            vgrid = grid + flow

            # scale grid to [-1,1]
            vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
            vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

            vgrid = vgrid.permute(0, 2, 3, 1)  # 处理后 shape 为 (B, H, W, 2)

            r = F.grid_sample(enc_h, vgrid, align_corners=True)
            warped_data.append(r)

        warped_data = torch.stack(warped_data).permute(1, 2, 3, 4, 0)

        return warped_data


class Decoder(nn.Module):
    def __init__(self, hidden_channels: int, k: int = 3):
        super(Decoder, self).__init__()
        self.conv_1x1 = nn.Conv2d(
            in_channels=hidden_channels // 4 * k ** 2, out_channels=hidden_channels // 4,
            kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)
        )

        self.deconv = nn.ConvTranspose2d(
            in_channels=hidden_channels // 4, out_channels=hidden_channels,
            kernel_size=(k, k), stride=(2, 2), padding=(1, 1), output_padding=(1, 1)
        )

    def forward(self, h_: Tensor):
        # h_ 的 shape 为 (B, C/4, H/2, W/2, k^2)
        batch = h_.shape[0]
        height = h_.shape[2]
        width = h_.shape[3]

        temp = h_.permute(0, 1, 4, 2, 3).reshape(batch, -1, height, width)
        temp = self.conv_1x1(temp)

        dec_h = self.deconv(temp)

        return dec_h

# if __name__ == '__main__':
#     batch = 3
#     height = 128
#     width = 128
#     hidden_channels = 32
#     k = 3
#     alpha = 0.5
#     gru = MotionGRU(hidden_channels=hidden_channels, k=k, alpha=alpha)
#     h = torch.ones(batch, hidden_channels, height, width)
#     f = torch.ones(batch, 2 * k ** 2, height // 2, width // 2)
#     d = torch.ones(batch, 2 * k ** 2, height // 2, width // 2)
#
#     x, f, d = gru(h, f, d)
#     print(x.shape)
#     print(f.shape)
#     print(d.shape)
