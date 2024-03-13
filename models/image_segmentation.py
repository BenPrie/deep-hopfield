# Imports as always.
import torch
from torch import nn

from segmentation_models_pytorch import UnetPlusPlus
from segmentation_models_pytorch.losses import DiceLoss

class DownConv2(nn.Module):
    def __init__(self, n_in_channels, n_out_channels, kernel_size, activation_func):
        super().__init__()

        self.seq = nn.Sequential(
            nn.Conv2d(
                in_channels=n_in_channels,
                out_channels=n_out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False
            ),
            nn.BatchNorm2d(n_out_channels),
            activation_func,

            nn.Conv2d(
                in_channels=n_out_channels,
                out_channels=n_out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False
            ),
            nn.BatchNorm2d(n_out_channels),
            activation_func
        )

        self.pool = nn.MaxPool2d(kernel_size=2, return_indices=True)

    def forward(self, x):
        y = self.seq(x)
        pool_shape = y.shape
        y, indices = self.pool(y)

        return y, indices, pool_shape


class DownConv3(nn.Module):
    def __init__(self, n_in_channels, n_out_channels, kernel_size, activation_func):
        super().__init__()

        self.seq = nn.Sequential(
            nn.Conv2d(
                in_channels=n_in_channels,
                out_channels=n_out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False
            ),
            nn.BatchNorm2d(n_out_channels),
            activation_func,

            nn.Conv2d(
                in_channels=n_out_channels,
                out_channels=n_out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False
            ),
            nn.BatchNorm2d(n_out_channels),
            activation_func,

            nn.Conv2d(
                in_channels=n_out_channels,
                out_channels=n_out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False
            ),
            nn.BatchNorm2d(n_out_channels),
            activation_func
        )

        self.pool = nn.MaxPool2d(kernel_size=2, return_indices=True)

    def forward(self, x):
        y = self.seq(x)
        pool_shape = y.shape
        y, indices = self.pool(y)

        return y, indices, pool_shape


class UpConv2(nn.Module):
    def __init__(self, n_in_channels, n_out_channels, kernel_size, activation_func):
        super().__init__()

        self.seq = nn.Sequential(
            nn.Conv2d(
                in_channels=n_in_channels,
                out_channels=n_in_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False
            ),
            nn.BatchNorm2d(n_in_channels),
            activation_func,

            nn.Conv2d(
                in_channels=n_in_channels,
                out_channels=n_out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False
            ),
            nn.BatchNorm2d(n_out_channels),
            activation_func
        )

        self.unpool = nn.MaxUnpool2d(kernel_size=2)

    def forward(self, x, indices, output_size):
        y = self.unpool(x, indices, output_size=output_size)
        y = self.seq(y)

        return y


class UpConv3(nn.Module):
    def __init__(self, n_in_channels, n_out_channels, kernel_size, activation_func):
        super().__init__()

        self.seq = nn.Sequential(
            nn.Conv2d(
                in_channels=n_in_channels,
                out_channels=n_in_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False
            ),
            nn.BatchNorm2d(n_in_channels),
            activation_func,

            nn.Conv2d(
                in_channels=n_in_channels,
                out_channels=n_in_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False
            ),
            nn.BatchNorm2d(n_in_channels),
            activation_func,

            nn.Conv2d(
                in_channels=n_in_channels,
                out_channels=n_out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False
            ),
            nn.BatchNorm2d(n_out_channels),
            activation_func
        )

        self.unpool = nn.MaxUnpool2d(kernel_size=2)

    def forward(self, x, indices, output_size):
        y = self.unpool(x, indices, output_size=output_size)
        y = self.seq(y)

        return y


class ImageSegmentationCNN(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()

        self.out_channels = 3
        self.activation = nn.ReLU()

        self.bn_input = nn.BatchNorm2d(3)

        self.dc1 = DownConv2(3, 64, kernel_size, self.activation)
        self.dc2 = DownConv2(64, 128, kernel_size, self.activation)
        self.dc3 = DownConv3(128, 256, kernel_size, self.activation)
        self.dc4 = DownConv3(256, 512, kernel_size, self.activation)

        self.uc4 = UpConv3(512, 256, kernel_size, self.activation)
        self.uc3 = UpConv3(256, 128, kernel_size, self.activation)
        self.uc2 = UpConv2(128, 64, kernel_size, self.activation)
        self.uc1 = UpConv2(64, 1, kernel_size, self.activation)

    def forward(self, batch: torch.Tensor):
        x = self.bn_input(batch)

        # Encoder.
        x, pool1_indices, shape1 = self.dc1(x)
        x, pool2_indices, shape2 = self.dc2(x)
        x, pool3_indices, shape3 = self.dc3(x)
        x, pool4_indices, shape4 = self.dc4(x)

        # Decoder.
        x = self.uc4(x, pool4_indices, shape4)
        x = self.uc3(x, pool3_indices, shape3)
        x = self.uc2(x, pool2_indices, shape2)
        x = self.uc1(x, pool1_indices, shape1)

        return x


class SegmentationModel(nn.Module):
    def __init__(self):
        super(SegmentationModel, self).__init__()

        self.model = UnetPlusPlus(
            encoder_name='timm-efficientnet-b0',
            encoder_weights='imagenet',
            in_channels=3,
            classes=1,
            activation=None
        )

    def forward(self, x, y_true):
        y_pred = self.model(x)

        if y_true is not None:
            loss = DiceLoss(mode='binary')(y_pred, y_true)
            return y_pred, loss

        return y_pred