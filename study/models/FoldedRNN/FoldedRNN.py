from dGRU import dGRU
import torch
from torch import nn
from UpSample import UpSample


class FoldedRNN(nn.Module):
    def __init__(self, in_channels, img_shape):
        super(FoldedRNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=(5, 5), stride=(1, 1)),
            nn.Tanh()
        )

        img_shape_1 = [edge - 4 for edge in img_shape]

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), stride=(1, 1)),
            nn.Tanh(),
        )

        img_shape_2 = [edge - 4 for edge in img_shape_1]

        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        img_shape_3 = [(edge - 2) // 2 + 1 for edge in img_shape_2]

        self.dGRU_1 = dGRU(in_channels=64, hidden_channels=128, kernel_size=5)
        self.dGRU_2 = dGRU(in_channels=128, hidden_channels=128, kernel_size=5)

        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        img_shape_4 = [(edge - 2) // 2 + 1 for edge in img_shape_3]

        self.dGRU_3 = dGRU(in_channels=128, hidden_channels=256, kernel_size=5)
        self.dGRU_4 = dGRU(in_channels=256, hidden_channels=256, kernel_size=5)

        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        img_shape_5 = [(edge - 2) // 2 + 1 for edge in img_shape_4]

        self.dGRU_5 = dGRU(in_channels=256, hidden_channels=512, kernel_size=3)
        self.dGRU_6 = dGRU(in_channels=512, hidden_channels=512, kernel_size=3)

        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.dGRU_7 = dGRU(in_channels=512, hidden_channels=256, kernel_size=3)
        self.dGRU_8 = dGRU(in_channels=256, hidden_channels=256, kernel_size=3)

        out_padding1 = []
        for x_in, x_out in zip(img_shape_1, img_shape_2):
            out_padding1.append(
                x_in - x_out - 4
            )

        out_padding2 = []
        for x_in, x_out in zip(img_shape_1, img_shape_2):
            out_padding2.append(
                x_in - x_out - 4
            )

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=32, out_channels=in_channels, kernel_size=(5, 5), stride=(1, 1),
                output_padding=tuple(out_padding1)
            ),
            nn.Tanh()
        )

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=64, out_channels=32, kernel_size=(5, 5), stride=(1, 1),
                output_padding=tuple(out_padding1)
            ),
            nn.Tanh(),
        )

        self.unpool1 = UpSample(img_shape=img_shape_2)
        self.unpool2 = UpSample(img_shape=img_shape_3)
        self.unpool3 = UpSample(img_shape=img_shape_4)
        self.unpool4 = UpSample(img_shape=img_shape_5)

    def _forward(self, x):
        """
        :param x:  (B,C,H,W)
        :return:
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.dGRU_1(x)
        x = self.dGRU_2(x)
        x = self.pool2(x)
        x = self.dGRU_3(x)
        x = self.dGRU_4(x)
        x = self.pool3(x)
        x = self.dGRU_5(x)
        x = self.dGRU_6(x)
        x = self.pool4(x)
        x = self.dGRU_7(x)
        x = self.dGRU_8(x)
        return x

    def _backward(self, x):
        x = self.dGRU_8.backward(x)
        x = self.dGRU_7.backward(x)
        x = self.unpool4(x)
        x = self.dGRU_6.backward(x)
        x = self.dGRU_5.backward(x)
        x = self.unpool3(x)
        x = self.dGRU_4.backward(x)
        x = self.dGRU_3.backward(x)
        x = self.unpool2(x)
        x = self.dGRU_2.backward(x)
        x = self.dGRU_1.backward(x)
        x = self.unpool1(x)
        x = self.deconv2(x)
        x = self.deconv1(x)
        return x

    # noinspection PyUnboundLocalVariable
    def forward(self, inputs, out_len=10):
        """
        :param inputs: (batch, sequence, channel, height, width)
        :param out_len:  外推长度
        :return:
        """
        batch, sequence, channel, height, width = inputs.shape

        pred = []  # 存储预测结果

        for s in range(sequence + out_len):
            if s < sequence:
                x = inputs[:, s]
            else:
                x = x_pred

            x = self._forward(x)
            x_pred = self._backward(x)

            if s >= sequence:
                pred.append(x_pred)

        # [length, batch, channel, height, width] -> [batch, length, channel, height, width]
        prediction = torch.stack(pred, dim=0).permute(1, 0, 2, 3, 4)

        return prediction


# if __name__ == '__main__':
#     device = "cuda"
#     folded_rnn = FoldedRNN(in_channels=1, img_shape=[64, 64]).to(device)
#     inputs = torch.ones(3, 10, 1, 64, 64).to(device)
#
#     result = folded_rnn(inputs)
#     print(result.shape)
#     result.sum().backward()
