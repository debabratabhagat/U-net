from . import config
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import ConvTranspose2d
from torch.nn import Conv2d, MaxPool2d
from torch.nn import ReLU

from torchvision.transforms import CenterCrop
from torch.nn import functional as F

import torch


class Block(Module):
    def __init__(self, inChannels, outChannels):
        super().__init__()

        self.conv1 = Conv2d(inChannels, outChannels, 3)
        self.relu = ReLU()
        self.conv2 = Conv2d(outChannels, outChannels, 3)

    def forward(self, x):
        layer1 = self.conv1(x)
        relu = self.relu(layer1)
        layer2 = self.conv2(relu)

        return layer2


class Encoder(Module):
    def __init__(self, channels=(3, 16, 32, 64)):
        super().__init__()

        self.encBlocks = ModuleList(
            [Block(channels[i], channels[i + 1]) for i in range(len(channels) - 1)]
        )
        self.pool = MaxPool2d(2)

    def forward(self, x):
        blockOutputs = []

        for block in self.encBlocks:
            x = block(x)
            blockOutputs.append(x)
            x = self.pool(x)

        return blockOutputs


class Decoder(Module):
    def __init__(self, channels=(64, 32, 16)):
        super().__init__()

        self.channels = channels
        self.upconvs = ModuleList(
            [
                ConvTranspose2d(self.channels[i], self.channels[i + 1], 2, 2)
                for i in range(len(self.channels) - 1)
            ]
        )

        self.decBlocks = ModuleList(
            [
                Block(self.channels[i], self.channels[i + 1])
                for i in range(len(self.channels) - 1)
            ]
        )

    def crop(self, encFeatures, x):
        _, _, h, w = x.shape
        encFeatures = CenterCrop([h, w])(encFeatures)
        return encFeatures

    def forward(self, x, encFeatures):
        for i in range(len(self.channels) - 1):
            x = self.upconvs[i](x)

            encFeats = self.crop(encFeatures[i], x)
            x = torch.cat([x, encFeats], dim=1)
            x = self.decBlocks[i](x)

        return x


class UNet(Module):
    def __init__(
        self,
        encChannels=(3, 16, 32, 64),
        decChannels=(64, 32, 16),
        nbClasses=1,
        retainDim=True,
        outSize=(config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH),
    ):
        super().__init__()
        self.encoder = Encoder(encChannels)
        self.decoder = Decoder(decChannels)

        self.head = Conv2d(decChannels[-1], nbClasses, 1)
        self.retainDim = retainDim
        self.outSize = outSize

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded[::-1][0], encoded[::-1][1:])

        map = self.head(decoded)

        if self.retainDim:
            map = F.interpolate(map, self.outSize)

        return map
