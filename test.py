# pylint: disable=no-member, invalid-name, missing-docstring
import torch
import d4cnn

d4cnn.group.test_group()
d4cnn.group.test_field_representation()

image = torch.randn(2, 8, 16, 64, 64)
d4cnn.conv.test_D4ConvRR(image, 16, 3)

image = torch.randn(2, 16, 64, 64)
d4cnn.conv.test_D4ConvIR(image, 16, 3)

image = torch.randn(2, 8, 16, 64, 64)
d4cnn.conv.test_D4ConvRI(image, 16, 3)
