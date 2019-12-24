# pylint: disable=no-member, invalid-name, missing-docstring, redefined-builtin, arguments-differ, line-too-long
import torch
import torch.nn as nn
import torch.nn.functional as F

from .group import field_action


class D4BatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super().__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.momentum = momentum

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        if self.affine:
            with torch.no_grad():
                self.weight.fill_(1)  # changed this (was uniform_ originally)
                self.bias.zero_()

    def forward(self, input):
        if input.dim() == 5:
            # input [batch, channel, repr, y, x]
            assert input.size(2) == 8

            # input [batch, channel, repr * y, x]
            output = input.view(input.size(0), input.size(1), input.size(2) * input.size(3), input.size(4))
        if input.dim() == 4:
            # input [batch, channel, y, x]
            output = input

        output = F.batch_norm(
            output, self.running_mean, self.running_var, self.weight, self.bias,
            self.training, self.momentum, self.eps)
        output = output.view(*input.size())
        return output

    def __repr__(self):
        return ('{name}({num_features}, eps={eps}, momentum={momentum},'
                ' affine={affine})'
                .format(name=self.__class__.__name__, **self.__dict__))


def test_D4BatchNorm2d(image, **kwargs):
    # image [batch, channel, repr, y, x]
    bn = D4BatchNorm2d(image.size(1), **kwargs)
    with torch.no_grad():
        bn.weight.normal_()
        bn.bias.normal_()

    bn.train()
    xs = [field_action(u, bn(image), 2, 3, 4) for u in range(8)]
    ys = [bn(field_action(u, image, 2, 3, 4)) for u in range(8)]

    for x, y in zip(xs, ys):
        r = (x - y).abs().max() / x.abs().max()
        assert r < 1e-5, repr(r)

    bn.eval()
    xs = [field_action(u, bn(image), 2, 3, 4) for u in range(8)]
    ys = [bn(field_action(u, image, 2, 3, 4)) for u in range(8)]

    for x, y in zip(xs, ys):
        r = (x - y).abs().max() / x.abs().max()
        assert r < 1e-5, repr(r)

    return xs, ys
