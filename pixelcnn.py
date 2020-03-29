import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MaskedConv2d(nn.Conv2d):
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


class PixelCNN(nn.Module):
    def __init__(self):
        super().__init__()
        num_filters = 64
        self.net = nn.Sequential(
            MaskedConv2d('A', 1, num_filters, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),

            MaskedConv2d('B', num_filters, num_filters, kernel_size=7, stride=1, padding=3), nn.ReLU(),
            nn.BatchNorm2d(num_filters),
            MaskedConv2d('B', num_filters, num_filters, kernel_size=7, stride=1, padding=3), nn.ReLU(),
            nn.BatchNorm2d(num_filters),
            MaskedConv2d('B', num_filters, num_filters, kernel_size=7, stride=1, padding=3), nn.ReLU(),
            nn.BatchNorm2d(num_filters),
            MaskedConv2d('B', num_filters, num_filters, kernel_size=7, stride=1, padding=3), nn.ReLU(),
            nn.BatchNorm2d(num_filters),
            MaskedConv2d('B', num_filters, num_filters, kernel_size=7, stride=1, padding=3), nn.ReLU(),
            nn.BatchNorm2d(num_filters),

            MaskedConv2d('B', num_filters, num_filters, kernel_size=1, stride=1, padding=0), nn.ReLU(),
            nn.BatchNorm2d(num_filters),
            MaskedConv2d('B', num_filters, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)