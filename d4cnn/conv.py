# pylint: disable=E1101,R,C
import torch
import torch.nn as nn
import torch.nn.functional as F
from .group import d4_inv, d4_mul, image_all_actions, field_all_actions


def uniform(*size):
    x = torch.rand(*size)
    return (2 * x - 1) * (3 ** 0.5)


class D4ConvRR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.weight = nn.Parameter(uniform(8, out_channels, in_channels, kernel_size, kernel_size))
        self.scale = (8 * in_channels * kernel_size ** 2) ** -0.5
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)

    def forward(self, input):  # pylint: disable=W
        # input [batch, repr, channel, y, x]
        assert input.dim() == 5
        assert input.size(1) == 8

        ws = image_all_actions(self.weight, 3, 4)

        weight = torch.cat([
            torch.cat([
                ws[w][d4_mul[d4_inv[w]][v]]
                for v in range(8)
            ], dim=1)
            for w in range(8)
        ], dim=0)

        output = input.view(input.size(0), 8 * input.size(2), input.size(3), input.size(4))
        output = F.conv2d(output, self.scale * weight, **self.kwargs)
        output = output.view(input.size(0), 8, -1, output.size(2), output.size(3))
        if self.bias is not None:
            output = output + self.bias.view(-1, 1, 1)  # TODO add bias via F.conv2d
        return output


def test_D4ConvRR(image, out_channels, kernel_size, **kwargs):
    # image [batch, repr, channel, y, x]
    c = D4ConvRR(image.size(2), out_channels, kernel_size, **kwargs)

    xs = field_all_actions(c(image), 1, 3, 4)
    ys = [c(gx) for gx in field_all_actions(image, 1, 3, 4)]

    for x, y in zip(xs, ys):
        r = (x - y).std() / x.std()
        assert r < 1e-5, repr(r)

    return xs, ys


class D4ConvIR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.weight = nn.Parameter(uniform(out_channels, in_channels, kernel_size, kernel_size))
        self.scale = (in_channels * kernel_size ** 2) ** -0.5
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)

    def forward(self, input):  # pylint: disable=W
        # input [batch, channel, y, x]
        assert input.dim() == 4

        ws = image_all_actions(self.weight, 2, 3)

        weight = torch.cat(ws, dim=0)

        output = F.conv2d(input, self.scale * weight, **self.kwargs)
        output = output.view(input.size(0), 8, -1, output.size(2), output.size(3))
        if self.bias is not None:
            output = output + self.bias.view(-1, 1, 1)
        return output


def test_D4ConvIR(image, out_channels, kernel_size, **kwargs):
    # image [batch, channel, y, x]
    c = D4ConvIR(image.size(1), out_channels, kernel_size, **kwargs)

    xs = field_all_actions(c(image), 1, 3, 4)
    ys = [c(gx) for gx in image_all_actions(image, 2, 3)]

    for x, y in zip(xs, ys):
        r = (x - y).std() / x.std()
        assert r < 1e-5, repr(r)

    return xs, ys


class D4ConvRI(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.weight = nn.Parameter(uniform(out_channels, in_channels, kernel_size, kernel_size))
        self.scale = (8 * in_channels * kernel_size ** 2) ** -0.5
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)

    def forward(self, input):  # pylint: disable=W
        # input [batch, repr, channel, y, x]
        assert input.dim() == 5
        assert input.size(1) == 8

        ws = image_all_actions(self.weight, 2, 3)

        weight = torch.cat(ws, dim=1)

        output = input.view(input.size(0), 8 * input.size(2), input.size(3), input.size(4))
        output = F.conv2d(output, self.scale * weight, self.bias, **self.kwargs)
        return output


def test_D4ConvRI(image, out_channels, kernel_size, **kwargs):
    # image [batch, repr, channel, y, x]
    c = D4ConvRI(image.size(2), out_channels, kernel_size, **kwargs)

    xs = image_all_actions(c(image), 2, 3)
    ys = [c(gx) for gx in field_all_actions(image, 1, 3, 4)]

    for x, y in zip(xs, ys):
        r = (x - y).std() / x.std()
        assert r < 1e-5, repr(r)

    return xs, ys
