# pylint: disable=E1101,R,C
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .group import d4_inv, d4_mul, image_all_actions, field_all_actions


class D4ConvRR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.weight = nn.Parameter(torch.randn(8, out_channels, in_channels, kernel_size, kernel_size))

    def forward(self, x):  # pylint: disable=W0221
        # x [batch, repr, channel, y, x]
        assert x.dim() == 5
        assert x.size(1) == 8

        ws = image_all_actions(self.weight, 3, 4)

        weight = torch.cat([
            torch.cat([
                ws[w][d4_mul[d4_inv[w]][v]]
                for v in range(8)
            ], dim=1)
            for w in range(8)
        ], dim=0)

        y = x.view(x.size(0), 8 * x.size(2), x.size(3), x.size(4))
        y = F.conv2d(y, weight, **self.kwargs)
        y = y.view(x.size(0), 8, -1, y.size(2), y.size(3))
        return y


def test_D4ConvRR(image, out_channels, kernel_size, padding):
    # image [batch, repr, channel, y, x]
    c = D4ConvRR(image.size(2), out_channels, kernel_size, padding=padding)

    xs = field_all_actions(c(Variable(image)).data, 1, 3, 4)
    ys = [c(Variable(gx)).data for gx in field_all_actions(image, 1, 3, 4)]

    for x, y in zip(xs, ys):
        assert (x - y).std() / x.std() < 1e-6

    return xs, ys


class D4ConvIR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))

    def forward(self, x):  # pylint: disable=W0221
        # x [batch, channel, y, x]
        assert x.dim() == 4

        ws = image_all_actions(self.weight, 2, 3)

        weight = torch.cat(ws, dim=0)

        y = F.conv2d(x, weight, **self.kwargs)
        y = y.view(x.size(0), 8, -1, y.size(2), y.size(3))
        return y


def test_D4ConvIR(image, out_channels, kernel_size, padding):
    # image [batch, channel, y, x]
    c = D4ConvIR(image.size(1), out_channels, kernel_size, padding=padding)

    xs = field_all_actions(c(Variable(image)).data, 1, 3, 4)
    ys = [c(Variable(gx)).data for gx in image_all_actions(image, 2, 3)]

    for x, y in zip(xs, ys):
        assert (x - y).std() / x.std() < 1e-6

    return xs, ys
