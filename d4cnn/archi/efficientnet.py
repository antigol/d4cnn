# pylint: disable=no-member, invalid-name, missing-docstring, redefined-builtin, arguments-differ, line-too-long
import torch
import torch.nn as nn

from d4cnn import D4BatchNorm2d, D4ConvIR, D4ConvRR, D4ConvRI


class GlobalAvgPool2d(nn.Module):
    def forward(self, x):
        return x.mean([-1, -2], keepdim=True)


def change_submodules(m, fn):
    for name in dir(m):
        sm = getattr(m, name)
        if isinstance(sm, torch.nn.Module):
            setattr(m, name, fn(sm))

    if isinstance(m, (torch.nn.Sequential, torch.nn.ModuleList)):
        for i, sm in enumerate(m):
            m[i] = fn(sm)

    for ch in m.children():
        change_submodules(ch, fn)


def equivariantize_module(m):
    class_name = m.__class__.__name__

    if class_name == "Conv2dSame":
        assert m.kernel_size[0] % 2 == 1
        padding = m.kernel_size[0] // 2
        return D4ConvRR(m.in_channels, m.out_channels, m.kernel_size[0],
                        stride=m.stride, padding=padding, dilation=m.dilation, bias=m.bias is not None, groups=m.groups)

    if class_name == "Conv2d":
        return D4ConvRR(m.in_channels, m.out_channels, m.kernel_size[0],
                        stride=m.stride, padding=m.padding, dilation=m.dilation, bias=m.bias is not None, groups=m.groups)

    if isinstance(m, nn.BatchNorm2d):
        return D4BatchNorm2d(m.num_features, m.eps, m.momentum, m.affine)

    if isinstance(m, nn.AdaptiveAvgPool2d):
        return GlobalAvgPool2d()

    return m


def equivariantize_network(f, in_channels):
    m = f.conv_stem
    padding = m.kernel_size[0] // 2
    f.conv_stem = D4ConvIR(in_channels, m.out_channels, m.kernel_size[0],
                           stride=m.stride, padding=padding, dilation=m.dilation, bias=m.bias is not None, groups=m.groups)

    #f.global_pool = D4Wrapper(f.global_pool)

    m = f.conv_head
    f.conv_head = D4ConvRI(m.in_channels, m.out_channels, m.kernel_size[0], m.bias is not None)

    change_submodules(f, equivariantize_module)

    return f
