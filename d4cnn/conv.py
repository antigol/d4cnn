# pylint: disable=no-member, invalid-name, missing-docstring, redefined-builtin, arguments-differ, line-too-long
import torch
import torch.nn as nn
import torch.nn.functional as F

from .group import d4_inv, d4_mul, field_all_actions, image_all_actions, field_action, image_action


def _from_tuple(x):
    if isinstance(x, (int, float)):
        return x
    assert len(x) == 2 and x[0] == x[1]
    return x[0]


def _valid_stride(w, p, k, s):
    return (w + 2 * p - k) % s == 0


def uniform(*size):
    x = torch.rand(*size)
    return (2 * x - 1) * (3 ** 0.5)


class D4ConvRR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, padding=0, groups=1, stride=1, **kwargs):
        super().__init__()
        assert in_channels % groups == 0
        assert out_channels % groups == 0
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _from_tuple(kernel_size)
        self.padding = _from_tuple(padding)
        self.groups = _from_tuple(groups)
        self.stride = _from_tuple(stride)
        self.kwargs = kwargs
        self.weight = nn.Parameter(uniform(8, out_channels, 1, in_channels // self.groups, 1, self.kernel_size, self.kernel_size))
        self.scale = (8 * in_channels // self.groups * self.kernel_size ** 2) ** -0.5
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

        if all(_valid_stride(w, self.padding, self.kernel_size, self.stride) for w in input.shape[3:]):
            return self._conv(input, weight, bias)

        convs = [self._conv(x, weight, bias) for x in field_all_actions(input, 2, 3, 4)[:4]]
        return sum(field_action(d4_inv[u], x, 2, 3, 4) for u, x in enumerate(convs)) / 4

    def _conv(self, input, weight, bias):
        output = input.view(input.size(0), input.size(1) * 8, input.size(3), input.size(4))
        output = F.conv2d(output, self.scale * weight, bias, padding=self.padding, groups=self.groups, stride=self.stride, **self.kwargs)
        output = output.view(input.size(0), -1, 8, output.size(2), output.size(3))
        return output

    def __repr__(self):
        return ('{name}(in_channels={in_channels}, out_channels={out_channels},'
                ' kernel_size={kernel_size}, stride={stride})'
                .format(name=self.__class__.__name__, **self.__dict__))


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
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, padding=0, groups=1, stride=1, **kwargs):
        super().__init__()
        assert in_channels % groups == 0
        assert out_channels % groups == 0
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _from_tuple(kernel_size)
        self.padding = _from_tuple(padding)
        self.groups = _from_tuple(groups)
        self.stride = _from_tuple(stride)
        self.kwargs = kwargs
        self.weight = nn.Parameter(uniform(out_channels, 1, in_channels // self.groups, self.kernel_size, self.kernel_size))
        self.scale = (in_channels // self.groups * self.kernel_size ** 2) ** -0.5
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

        if all(_valid_stride(w, self.padding, self.kernel_size, self.stride) for w in input.shape[2:]):
            return self._conv(input, weight, bias)

        convs = [self._conv(x, weight, bias) for x in image_all_actions(input, 2, 3)[:4]]
        return sum(field_action(d4_inv[u], x, 2, 3, 4) for u, x in enumerate(convs)) / 4

    def _conv(self, input, weight, bias):
        output = F.conv2d(input, self.scale * weight, bias, padding=self.padding, groups=self.groups, stride=self.stride, **self.kwargs)
        output = output.view(input.size(0), -1, 8, output.size(2), output.size(3))
        return output

    def __repr__(self):
        return ('{name}(in_channels={in_channels}, out_channels={out_channels},'
                ' kernel_size={kernel_size} stride={stride})'
                .format(name=self.__class__.__name__, **self.__dict__))


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
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, padding=0, groups=1, stride=1, **kwargs):
        super().__init__()
        assert in_channels % groups == 0
        assert out_channels % groups == 0
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _from_tuple(kernel_size)
        self.padding = _from_tuple(padding)
        self.groups = _from_tuple(groups)
        self.stride = _from_tuple(stride)
        self.kwargs = kwargs
        self.weight = nn.Parameter(uniform(out_channels, in_channels // self.groups, 1, self.kernel_size, self.kernel_size))
        self.scale = (8 * in_channels // self.groups * self.kernel_size ** 2) ** -0.5
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

        if all(_valid_stride(w, self.padding, self.kernel_size, self.stride) for w in input.shape[3:]):
            return self._conv(input, weight, self.bias)

        convs = [self._conv(x, weight, self.bias) for x in field_all_actions(input, 2, 3, 4)[:4]]
        return sum(image_action(d4_inv[u], x, 2, 3) for u, x in enumerate(convs)) / 4

    def _conv(self, input, weight, bias):
        output = input.view(input.size(0), input.size(1) * 8, input.size(3), input.size(4))
        output = F.conv2d(output, self.scale * weight, bias, padding=self.padding, groups=self.groups, stride=self.stride, **self.kwargs)
        return output

    def __repr__(self):
        return ('{name}(in_channels={in_channels}, out_channels={out_channels},'
                ' kernel_size={kernel_size} stride={stride})'
                .format(name=self.__class__.__name__, **self.__dict__))


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
