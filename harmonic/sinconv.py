import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from coordconv import CoordConv, AddCoords

'''
An alternative implementation for PyTorch with auto-infering the x-y dimensions.
'''


class AddSine(AddCoords):
    def __init__(self, alpha=0.5, beta=None, phase_shift=0.):
        super(AddSine, self).__init__(False)
        if beta is None:
            beta = alpha
        self.alpha = Parameter(torch.FloatTensor([alpha]))
        self.beta = Parameter(torch.FloatTensor([beta]))
        self.phase = Parameter(torch.FloatTensor([phase_shift]))

    def generate_xy(self, input_tensor):
        batch_size, _, x_dim, y_dim = input_tensor.size()

        sx = self.phase

        xx_channel = torch.linspace(0., 1., x_dim).repeat(1, y_dim, 1).to(self.phase.device)
        yy_channel = torch.linspace(0., 1., y_dim).repeat(1, x_dim, 1).transpose(1, 2).to(self.phase.device)

        xx_channel = xx_channel.float() * self.alpha
        yy_channel = yy_channel.float() * self.beta

        channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3) + yy_channel.repeat(batch_size, 1, 1,
                                                                                             1).transpose(2, 3)
        channel = torch.sin(channel + sx)
        return channel

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
                                         "alpha=" + str(self.alpha.item()) + \
               " beta=" + str(self.beta.item()) + \
               " phase=" + str(self.phase.item()) + ")"

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """

        xx_channel = self.generate_xy(input_tensor)
        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor)], dim=1)

        return ret


class SinConv(CoordConv):
    def __init__(self, in_channels, out_channels, sins=[], **kwargs):
        super(SinConv, self).__init__(in_channels, out_channels, **kwargs)
        self.addcoords = nn.Sequential(*[AddSine(a, b, p) for a, b, p in sins])
        in_size = in_channels + len(sins)
        self.conv = nn.Conv2d(in_size, out_channels, **kwargs)
