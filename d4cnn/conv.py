# pylint: disable=no-member, invalid-name, missing-docstring, redefined-builtin, arguments-differ
import torch
import torch.nn as nn
import torch.nn.functional as F

from .group import d4_inv, d4_mul, field_all_actions, image_all_actions


def uniform(*size):
    x = torch.rand(*size)
    return (2 * x - 1) * (3 ** 0.5)


class D4ConvRR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, groups=1, **kwargs):
        super().__init__()
        assert in_channels % groups == 0
        assert out_channels % groups == 0
        self.kwargs = kwargs
        self.weight = nn.Parameter(uniform(8, out_channels, 1, in_channels // groups, 1, kernel_size, kernel_size))
        self.scale = (8 * in_channels // groups * kernel_size ** 2) ** -0.5
        self.groups = groups
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)

    def forward(self, input):
        # input [batch, channel, repr, y, x]
        assert input.dim() == 5
        assert input.size(2) == 8

        ws = image_all_actions(self.weight, 5, 6)

        weight = torch.cat([
            torch.cat([
                ws[w][d4_mul[d4_inv[w]][v]]
                for v in range(8)
            ], dim=3)
            for w in range(8)
        ], dim=1)
        weight = weight.view(weight.size(0) * 8, weight.size(2) * 8, weight.size(4), weight.size(5))

        bias = None
        if self.bias is not None:
            bias = self.bias.view(-1, 1).repeat(1, 8).view(-1)

        output = input.view(input.size(0), input.size(1) * 8, input.size(3), input.size(4))
        output = F.conv2d(output, self.scale * weight, bias, groups=self.groups, **self.kwargs)
        output = output.view(input.size(0), -1, 8, output.size(2), output.size(3))
        return output


def test_D4ConvRR(image, out_channels, kernel_size, **kwargs):
    # image [batch, channel, repr, y, x]
    c = D4ConvRR(image.size(1), out_channels, kernel_size, **kwargs)
    with torch.no_grad():
        c.bias.normal_()

    xs = field_all_actions(c(image), 2, 3, 4)
    ys = [c(gx) for gx in field_all_actions(image, 2, 3, 4)]

    for x, y in zip(xs, ys):
        r = (x - y).abs().max() / x.abs().max()
        assert r < 1e-5, repr(r)

    return xs, ys


class D4ConvIR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, groups=1, **kwargs):
        super().__init__()
        assert in_channels % groups == 0
        assert out_channels % groups == 0
        self.kwargs = kwargs
        self.weight = nn.Parameter(uniform(out_channels, 1, in_channels // groups, kernel_size, kernel_size))
        self.scale = (in_channels // groups * kernel_size ** 2) ** -0.5
        self.groups = groups
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)

    def forward(self, input):
        # input [batch, channel, y, x]
        assert input.dim() == 4

        ws = image_all_actions(self.weight, 3, 4)
        weight = torch.cat(ws, dim=1)
        weight = weight.view(weight.size(0) * 8, weight.size(2), weight.size(3), weight.size(4))

        bias = None
        if self.bias is not None:
            bias = self.bias.view(-1, 1).repeat(1, 8).view(-1)

        output = F.conv2d(input, self.scale * weight, bias, groups=self.groups, **self.kwargs)
        output = output.view(input.size(0), -1, 8, output.size(2), output.size(3))
        return output


def test_D4ConvIR(image, out_channels, kernel_size, **kwargs):
    # image [batch, channel, y, x]
    c = D4ConvIR(image.size(1), out_channels, kernel_size, **kwargs)
    with torch.no_grad():
        c.bias.normal_()

    xs = field_all_actions(c(image), 2, 3, 4)
    ys = [c(gx) for gx in image_all_actions(image, 2, 3)]

    for x, y in zip(xs, ys):
        r = (x - y).abs().max() / x.abs().max()
        assert r < 1e-5, repr(r)

    return xs, ys


class D4ConvRI(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, groups=1, **kwargs):
        super().__init__()
        assert in_channels % groups == 0
        assert out_channels % groups == 0
        self.kwargs = kwargs
        self.weight = nn.Parameter(uniform(out_channels, in_channels // groups, 1, kernel_size, kernel_size))
        self.scale = (8 * in_channels // groups * kernel_size ** 2) ** -0.5
        self.groups = groups
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)

    def forward(self, input):
        # input [batch, channel, repr, y, x]
        assert input.dim() == 5
        assert input.size(2) == 8

        ws = image_all_actions(self.weight, 3, 4)
        weight = torch.cat(ws, dim=2)
        weight = weight.view(weight.size(0), weight.size(1) * 8, weight.size(3), weight.size(4))

        output = input.view(input.size(0), input.size(1) * 8, input.size(3), input.size(4))
        output = F.conv2d(output, self.scale * weight, self.bias, groups=self.groups, **self.kwargs)
        return output


def test_D4ConvRI(image, out_channels, kernel_size, **kwargs):
    # image [batch, channel, repr, y, x]
    c = D4ConvRI(image.size(1), out_channels, kernel_size, **kwargs)
    with torch.no_grad():
        c.bias.normal_()

    xs = image_all_actions(c(image), 2, 3)
    ys = [c(gx) for gx in field_all_actions(image, 2, 3, 4)]

    for x, y in zip(xs, ys):
        r = (x - y).abs().max() / x.abs().max()
        assert r < 1e-5, repr(r)

    return xs, ys
