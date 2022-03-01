import torch
from torch import nn, Tensor

__all__ = ["SEBlock"]


class SEBlock(nn.Module):
    def __init__(self, channels: int):
        super(SEBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(1, 1),
                      stride=(1, 1), padding=(0, 0))
        )

    def forward(self, x: Tensor):
        reassigned_weights = self.fc(x)
        x = x * reassigned_weights
        return x


# if __name__ == '__main__':
#     a = torch.ones(3, 512, 4, 4)
#     net = SEBlock(channels=512)
#     r = net(a)
#     print(r.shape)
