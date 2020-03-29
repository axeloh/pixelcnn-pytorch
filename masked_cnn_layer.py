import torch
import torch.nn as nn

class MaskedConv2d(nn.Conv2d):
    """Regular Masked Convolutional Layer"""
    def __init__(self, mask_type, in_channels, out_channels, kernel_size, stride=1, padding=0, use_cuda=True):
        super(MaskedConv2d, self).__init__(in_channels, out_channels, kernel_size, padding=padding)
        # self.register_buffer('mask', self.weight.data.clone())#torch.ones(out_channels, in_channels, kernel_size, kernel_size))
        # self.mask.fill_(1)
        kH, kW = kernel_size, kernel_size
        self.device = torch.device('cuda') if use_cuda else None
        self.mask = torch.ones(out_channels, in_channels, kH, kW).to(self.device)
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)


class ResidualMaskedConv2d(nn.Module):
    """Residual Masked Conv. layer"""
    def __init__(self, input_dim):
        super().__init__()

        self.net = nn.Sequential(
            MaskedConv2d('B', input_dim, input_dim // 2, kernel_size=1),
            nn.ReLU(),
            MaskedConv2d('B', input_dim // 2, input_dim // 2, kernel_size=7, padding=3),
            nn.ReLU(),
            MaskedConv2d('B', input_dim // 2, input_dim, kernel_size=1)
        )

    def forward(self, x):
        return self.net(x) + x


class AutoregressiveMaskedConv2d(nn.Conv2d):
    """ Code inspired by: https://github.com/anordertoreclaim/PixelCNN/blob/master/pixelcnn/conv_layers.py """

    def __init__(self, mask_type, in_channels, out_channels, kernel_size, data_channels=3, padding=0):
        super(AutoregressiveMaskedConv2d, self).__init__(in_channels, out_channels, kernel_size, padding=padding)
        mask = np.zeros(self.weight.size(), dtype=np.float32)
        kH, kW = kernel_size, kernel_size

        mask[:, :, :kH//2, :] = 1
        mask[:, :, kH//2, :kW//2 + 1] = 1

        def color_mask(color_out, color_in):
            a = (np.arange(out_channels) % data_channels == color_out)[:, None]
            b = (np.arange(in_channels) % data_channels == color_in)[None, :]
            return a * b

        for out_channel in range(data_channels):
            for in_channel in range(out_channel + 1, data_channels):
                mask[color_mask(out_channel, in_channel), kH//2, kW//2] = 0

        if mask_type == 'A': # Center to be zero
            for channel in range(data_channels):
                mask[color_mask(channel, channel), kH//2, kW//2] = 0

        self.register_buffer('mask', torch.from_numpy(mask))

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class AutoregressiveResidualMaskedConv2d(nn.Module):
    """
    Residual Links between MaskedConv2d-layers
    As described in Figure 5 in "Pixel Recurrent Neural Networks" by Aaron van den Oord et. al.
    """
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            AutoregressiveMaskedConv2d('B', in_dim, in_dim//2, kernel_size=1, padding=0), nn.ReLU(),
            AutoregressiveMaskedConv2d('B', in_dim//2, in_dim//2, kernel_size=7, padding=3), nn.ReLU(),
            AutoregressiveMaskedConv2d('B', in_dim//2, in_dim, kernel_size=1, padding=0), nn.ReLU())

    def forward(self, x):
        return self.net(x) + x


class ConditionalMaskedConv2d(MaskedConv2d):
    """ Class extending nn.Conv2d to use masks and condition on class """

    def __init__(self, mask_type, num_classes, in_channels, out_channels, kernel_size, padding=0):
        super(ConditionalMaskedConv2d, self).__init__(mask_type, in_channels, out_channels, kernel_size,
                                                      padding=padding)
        self.V = nn.Parameter(torch.randn(out_channels, num_classes))

    def forward(self, x, class_condition):
        conv_output = super().forward(x)
        s = conv_output.shape
        conv_output = conv_output.view(s[0], s[1], s[2] * s[3]) + (self.V @ class_condition.T).T.unsqueeze(-1)
        return conv_output.reshape(s)