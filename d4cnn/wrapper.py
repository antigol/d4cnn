# pylint: disable=no-member, invalid-name, missing-docstring, redefined-builtin, arguments-differ
import torch.nn as nn


class D4Wrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, input):
        # input [batch, channel, repr, y, x]
        assert input.dim() == 5
        assert input.size(2) == 8

        output = input.view(input.size(0), input.size(1) * 8, input.size(3), input.size(4))
        output = self.module(output)
        output = output.view(input.size(0), -1, 8, output.size(2), output.size(3))
        return output
