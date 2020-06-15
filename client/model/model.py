import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from dropblock import DropBlock3D, LinearScheduler

config = {
    "act_fn": lambda: nn.LeakyReLU(0.1, inplace=True),
    "norm_fn": lambda c: nn.BatchNorm3d(num_features=c)
}

def densenet3d(snapshot=None):
    model = DenseNet()
    # model = DataParallel(model)

    if snapshot is None:
        initialize(model.modules())
        print("Random initialized")
    else:
        state_dict = torch.load(snapshot)
        model.load_state_dict(state_dict)
        print("loads weight from {}".format(snapshot))

    return model


def initialize(modules):
    for module in modules:
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.ConvTranspose3d):
            nn.init.kaiming_uniform_(module.weight, mode="fan_in")
        elif isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, mode="fan_in")
            module.bias.data.zero_()


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels):
        super(ConvBlock, self).__init__()
        growth_rate = 32
        bottleneck = 4
        act_fn = config["act_fn"]
        norm_fn = config["norm_fn"]

        self.in_channels = in_channels
        self.growth_rate = growth_rate

        self.add_module("norm_1", norm_fn(in_channels))
        self.add_module("act_1", act_fn())
        self.add_module("conv_1", nn.Conv3d(in_channels, bottleneck * growth_rate,
                                            kernel_size=1, stride=1, padding=0, bias=True))
        self.add_module("norm_2", norm_fn(bottleneck * growth_rate))
        self.add_module("act_2", act_fn())
        self.add_module("conv_2", nn.Conv3d(bottleneck * growth_rate, growth_rate,
                                            kernel_size=3, stride=1, padding=1, bias=True))

    def forward(self, x):
        super_forward = super(ConvBlock, self).forward
        residual = x
        # features = super_forward(x)
        features = checkpoint(super_forward, x)
        out = torch.cat([residual, features], 1)
        return out

    @property
    def out_channels(self):
        return self.in_channels + self.growth_rate


class TransmitBlock(nn.Sequential):
    def __init__(self, in_channels, is_last_layer):
        super(TransmitBlock, self).__init__()
        act_fn = config["act_fn"]
        norm_fn = config["norm_fn"]
        compression = 2

        assert in_channels % compression == 0

        self.in_channels = in_channels
        self.compression = compression

        self.add_module("norm", norm_fn(in_channels))
        self.add_module("act", act_fn())

        if not is_last_layer:
            self.add_module("conv", nn.Conv3d(in_channels, in_channels // compression,
                                              kernel_size=1, stride=1, padding=0, bias=True))
            self.add_module("pool", nn.AvgPool3d(kernel_size=2, stride=2, padding=0))
        else:
            self.compression = 1

    @property
    def out_channels(self):
        return self.in_channels // self.compression


class DenseNet(nn.Module):
    def __init__(self):
        super(DenseNet, self).__init__()
        input_channels = 1
        conv_channels = 32
        down_structure = [2, 2, 2]
        output_channels = 4
        act_fn = config["act_fn"]
        norm_fn = config["norm_fn"]
        self.features = nn.Sequential()
        self.features.add_module("init_conv", nn.Conv3d(input_channels, conv_channels,
                                                        kernel_size=3, stride=1, padding=1, bias=True))
        self.features.add_module("init_norm", norm_fn(conv_channels))
        self.features.add_module("init_act", act_fn())
        self.dropblock = LinearScheduler(
            DropBlock3D(drop_prob=0., block_size=5),
            start_value=0.,
            stop_value=0.5,
            nr_steps=5e3
        )

        channels = conv_channels
        self.features.add_module('drop_block', DropBlock3D(drop_prob=0.1, block_size=5))

        for i, num_layers in enumerate(down_structure):
            for j in range(num_layers):
                conv_layer = ConvBlock(channels)
                self.features.add_module("block{}_layer{}".format(i + 1, j + 1), conv_layer)
                channels = conv_layer.out_channels

            # down-sample
            trans_layer = TransmitBlock(channels, is_last_layer=(i == len(down_structure) - 1))
            self.features.add_module("transition{}".format(i + 1), trans_layer)
            channels = trans_layer.out_channels

        self.classifier = nn.Linear(channels, output_channels)

    def forward(self, x, **return_opts):
        self.dropblock.step()
        batch_size, _, z, h, w = x.size()

        features = self.dropblock(self.features(x))
        # print("features", features.size())
        pooled = F.adaptive_avg_pool3d(features, 1).view(batch_size, -1)
        # print("pooled", pooled.size())
        scores = self.classifier(pooled)
        # print("scored", scores.size())

        if len(return_opts) == 0:
            return scores
