# pylint: disable=E1101,R,C
import torch
import torch.nn as nn
import torch.nn.functional as F


class D4BatchNorm2d(nn.Module):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super().__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.momentum = momentum
        if self.affine:
            self.weight = nn.Parameter(torch.empty(num_features))
            self.bias = nn.Parameter(torch.empty(num_features))
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

    def forward(self, input):  # pylint: disable=W
        # input [batch, repr, channel, y, x]
        self._check_input_dim(input)
        output = input.view(input.size(0) * input.size(1), *input.size()[2:])  # input [batch * repr, channel, y, x]
        output = F.batch_norm(
            output, self.running_mean, self.running_var, self.weight, self.bias,
            self.training, self.momentum, self.eps)
        output = output.view(*input.size())
        return output

    def __repr__(self):
        return ('{name}({num_features}, eps={eps}, momentum={momentum},'
                ' affine={affine})'
                .format(name=self.__class__.__name__, **self.__dict__))

    def _check_input_dim(self, input):  # pylint: disable=W
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))
        if input.size(1) != 8:
            raise ValueError('expected size(1) = 8 (got {})'
                             .format(input.size(1)))
