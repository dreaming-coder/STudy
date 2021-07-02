from typing import List
import torch
import torch.nn.functional as F
from torch import nn


class UpSample(nn.Module):

    def __init__(self, img_shape: List[int]):
        super(UpSample, self).__init__()
        self.img_shape = img_shape

    def forward(self, x):
        return F.interpolate(x, size=self.img_shape,mode='nearest')


if __name__ == '__main__':
    up = UpSample(img_shape=[64, 64])
    a = torch.ones(2, 11, 32, 32)
    r = up(a)
    print(r.shape)
