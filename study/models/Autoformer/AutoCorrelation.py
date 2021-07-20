import torch
from torch import nn
from torch.fft import fft, ifft, fft2, ifft2


class AutoCorrelation(nn.Module):
    """
    该函数针对单头处理
    """

    def __init__(self):
        super(AutoCorrelation, self).__init__()

    def forward(self, q, k, v):
        """
        :param q:  (batch,  1, length)
        :param k:  (batch,  1, length)
        :param v:  (batch,  1, length)
        :return:
        """
        _, _, L = q.shape
        S_xx = fft(q) * torch.conj(fft(k))
        t_list = []
        for t in range(L):
            t_list.append(ifft(S_xx))
        pass


if __name__ == '__main__':
    a = torch.randn((3, 5, 25))
    net = AutoCorrelation()
    net(a, a, a)
