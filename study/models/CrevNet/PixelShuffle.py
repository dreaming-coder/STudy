from torch import nn, Tensor

__all__ = ["PixelShuffle"]


class PixelShuffle(nn.Module):
    def __init__(self, n: int = 2):
        super(PixelShuffle, self).__init__()
        self.n = n

    def inverse(self, inputs: Tensor):
        batch, channel, height, width = inputs.shape
        result = inputs.reshape(
            batch, channel // self.n ** 2, self.n, self.n, height, width
        ).permute(
            0, 1, 4, 2, 5, 3
        ).reshape(
            batch, -1, height * self.n, width * self.n
        )
        return result

    def forward(self, inputs: Tensor):
        batch, channel, height, width = inputs.shape
        assert height % self.n == 0 and width % self.n == 0, "尺寸无法整除"
        height //= self.n
        width //= self.n

        result = inputs.reshape(
            batch, channel, height, self.n, width, self.n
        ).permute(
            0, 1, 3, 5, 2, 4
        ).reshape(
            batch, channel * self.n ** 2, height, width
        )

        return result
