import torch
from torch import nn, Tensor

from ResidualConv2d import ResidualConv2d
from SEBlock import SEBlock

__all__ = ["FURENet"]


class FURENet(nn.Module):
    def __init__(self, in_channels: int):
        super(FURENet, self).__init__()
        self.z_downSample_operate_1_to_2 = ResidualConv2d(in_channels=in_channels, out_channels=64)
        self.z_downSample_operate_2_to_3 = ResidualConv2d(in_channels=64, out_channels=128)
        self.z_downSample_operate_3_to_4 = ResidualConv2d(in_channels=128, out_channels=192)
        self.z_downSample_operate_4_to_5 = ResidualConv2d(in_channels=192, out_channels=256)
        self.z_downSample_operate_5_to_6 = ResidualConv2d(in_channels=256, out_channels=384)
        self.z_downSample_operate_6_to_7 = ResidualConv2d(in_channels=384, out_channels=512)

        self.zdr_downSample_operate_1_to_2 = ResidualConv2d(in_channels=in_channels, out_channels=64)
        self.zdr_downSample_operate_2_to_3 = ResidualConv2d(in_channels=64, out_channels=128)
        self.zdr_downSample_operate_3_to_4 = ResidualConv2d(in_channels=128, out_channels=192)
        self.zdr_downSample_operate_4_to_5 = ResidualConv2d(in_channels=192, out_channels=256)
        self.zdr_downSample_operate_5_to_6 = ResidualConv2d(in_channels=256, out_channels=384)
        self.zdr_downSample_operate_6_to_7 = ResidualConv2d(in_channels=384, out_channels=512)

        self.kdp_downSample_operate_1_to_2 = ResidualConv2d(in_channels=in_channels, out_channels=64)
        self.kdp_downSample_operate_2_to_3 = ResidualConv2d(in_channels=64, out_channels=128)
        self.kdp_downSample_operate_3_to_4 = ResidualConv2d(in_channels=128, out_channels=192)
        self.kdp_downSample_operate_4_to_5 = ResidualConv2d(in_channels=192, out_channels=256)
        self.kdp_downSample_operate_5_to_6 = ResidualConv2d(in_channels=256, out_channels=384)
        self.kdp_downSample_operate_6_to_7 = ResidualConv2d(in_channels=384, out_channels=512)

        self.se_block = SEBlock(channels=512 * 3)

        self.upSample_7_to_6 = ResidualConv2d(in_channels=512 * 3, out_channels=384)
        self.upSample_6_to_5 = ResidualConv2d(in_channels=384 * 4, out_channels=256)
        self.upSample_5_to_4 = ResidualConv2d(in_channels=256 * 4, out_channels=192)
        self.upSample_4_to_3 = ResidualConv2d(in_channels=192 * 4, out_channels=128)
        self.upSample_3_to_2 = ResidualConv2d(in_channels=128 * 4, out_channels=64)
        self.upSample_2_to_1 = ResidualConv2d(in_channels=64 * 4, out_channels=in_channels)

    def forward(self, z: Tensor, zdr: Tensor, kdp: Tensor):
        z2 = self.z_downSample_operate_1_to_2(z)
        z3 = self.z_downSample_operate_2_to_3(z2)
        z4 = self.z_downSample_operate_3_to_4(z3)
        z5 = self.z_downSample_operate_4_to_5(z4)
        z6 = self.z_downSample_operate_5_to_6(z5)
        z7 = self.z_downSample_operate_6_to_7(z6)

        zdr2 = self.zdr_downSample_operate_1_to_2(zdr)
        zdr3 = self.zdr_downSample_operate_2_to_3(zdr2)
        zdr4 = self.zdr_downSample_operate_3_to_4(zdr3)
        zdr5 = self.zdr_downSample_operate_4_to_5(zdr4)
        zdr6 = self.zdr_downSample_operate_5_to_6(zdr5)
        zdr7 = self.zdr_downSample_operate_6_to_7(zdr6)

        kdp2 = self.kdp_downSample_operate_1_to_2(kdp)
        kdp3 = self.kdp_downSample_operate_2_to_3(kdp2)
        kdp4 = self.kdp_downSample_operate_3_to_4(kdp3)
        kdp5 = self.kdp_downSample_operate_4_to_5(kdp4)
        kdp6 = self.kdp_downSample_operate_5_to_6(kdp5)
        kdp7 = self.kdp_downSample_operate_6_to_7(kdp6)

        concat_7 = torch.cat([zdr7, kdp7, z7], dim=1)
        upSampleInput = self.se_block(concat_7)

        upSample6 = self.upSample_7_to_6(upSampleInput)
        concat_6 = torch.cat([zdr6, kdp6, z6, upSample6], dim=1)

        upSample5 = self.upSample_6_to_5(concat_6)
        concat_5 = torch.cat([zdr5, kdp5, z5, upSample5], dim=1)

        upSample4 = self.upSample_5_to_4(concat_5)
        concat_4 = torch.cat([zdr4, kdp4, z4, upSample4], dim=1)

        upSample3 = self.upSample_4_to_3(concat_4)
        concat_3 = torch.cat([zdr3, kdp3, z3, upSample3], dim=1)

        upSample2 = self.upSample_3_to_2(concat_3)
        concat_2 = torch.cat([zdr2, kdp2, z2, upSample2], dim=1)

        out = self.upSample_2_to_1(concat_2)

        return out


if __name__ == '__main__':
    z = torch.ones(3, 10, 256, 256)
    zdr = torch.ones(3, 10, 256, 256)
    kdp = torch.ones(3, 10, 256, 256)
    net = FURENet(10)
    r = net(z, zdr, kdp)
    print(r.shape)
