import torch
import torch.nn as nn

'''
An alternative implementation for PyTorch with auto-infering the x-y dimensions.
From https://github.com/mkocabas/CoordConv-pytorch
'''


class AddCoords(nn.Module):
    def __init__(self, normalize=True):
        super(AddCoords, self).__init__()
        self.normalize = normalize

    def generate_xy(self, input_tensor):
        batch_size, _, x_dim, y_dim = input_tensor.size()

        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        if self.normalize:
            xx_channel = xx_channel.float() / (x_dim - 1)
            yy_channel = yy_channel.float() / (y_dim - 1)

            xx_channel = xx_channel * 2 - 1
            yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        return xx_channel, yy_channel

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """

        xx_channel, yy_channel = self.generate_xy(input_tensor)
        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)

        return ret


class CoordConv(nn.Module):
    def __init__(self, in_channels, out_channels, sins=[], **kwargs):
        super(CoordConv, self).__init__()
        self.addcoords = AddCoords()
        in_size = in_channels + 2
        self.conv = nn.Conv2d(in_size, out_channels, **kwargs)

    def forward(self, x):
        ret = self.addcoords(x)
        ret = self.conv(ret)
        return ret
