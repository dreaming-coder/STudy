import torch
from torch import nn


class Projector(nn.Module):
    def __init__(self, length: int):
        super(Projector, self).__init__()
        self.w_Q = nn.Linear(length, length)
        self.w_K = nn.Linear(length, length)
        self.w_V = nn.Linear(length, length)

    def forward(self, X):
        Q = torch.relu(self.w_Q(X))
        K = torch.relu(self.w_Q(X))
        V = torch.relu(self.w_Q(X))
        return Q, K, V

# if __name__ == '__main__':
#     x = torch.ones(3, 7, 30)
#     net = Projector(30)
#     r = net(x)
#     print(r[0].shape)
