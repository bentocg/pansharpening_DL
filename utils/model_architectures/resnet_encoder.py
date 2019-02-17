from torch import nn


def conv(ni, no): return nn.Conv2d(ni, no, kernel_size=3, stride=1, padding=0)


def conv2(ni, no): return nn.Conv2d(ni, no, kernel_size=3, stride=2, padding=0)


def conv_layer(ni, no):
    return nn.Sequential(nn.ReflectionPad2d(1), conv(ni, no), nn.BatchNorm2d(no),
                         nn.LeakyReLU(inplace=True))


def conv_layer2(ni, no):
    return nn.Sequential(nn.ReflectionPad2d(1), conv2(ni, no), nn.BatchNorm2d(no),
                         nn.LeakyReLU(inplace=True))


class ResBlock(nn.Module):
    def __init__(self, ni):
        super().__init__()
        self.conv1 = conv_layer(ni, ni)
        self.conv2 = conv_layer(ni, ni)

    def forward(self, x):
        return x.add_(self.conv2(self.conv1(x)))


def conv_and_res(ni, no, drop=0):
    return nn.Sequential(conv_layer2(ni, no), ResBlock(no), nn.Dropout(drop, inplace=True))



def resnet_encoder(n_channels, blocks=(16, 64, 256, 1028), final_dim=32, drop=0.1):
    return nn.Sequential(conv_and_res(n_channels, blocks[0]),
                         conv_and_res(blocks[0], blocks[1]),
                         conv_and_res(blocks[1], blocks[2]),
                         conv_and_res(blocks[2], blocks[3]),
                         nn.AdaptiveAvgPool2d(final_dim))