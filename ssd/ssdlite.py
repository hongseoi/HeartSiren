import torch
from torch import nn
from torch.autograd import Variable

from layers import *
from layers.modules.mobilenetv3layers import *
from layers.modules.mobilenetv3 import *
from layers.modules.multiboxlayer import MultiBoxLayer
from data.config import circor

# Inspired by https://github.com/amdegroot/ssd.pytorch/ and https://github.com/qfgaohao/pytorch-ssd/


class SSDLite(nn.Module):
    """SSD Lite

    Uses a MobileNetV3 passed as base_net as the backbone network.
    """

    def __init__(self, phase: str, n_classes: int):
        super(SSDLite, self).__init__()

        self._phase = phase
        self._n_classes = n_classes
        self._base_layers = nn.ModuleList()
        self.cfg = circor
        self.priorbox = PriorBox(self.cfg)
        self.priors = Variable(self.priorbox.forward(), volatile=True)

        # [kernel_size, hidden_channels(exp size), in_channels, out_channels(#out), SE, NL, s]
        self._block_layer_configs = [[3,   16,  16,  16, False, 'RE', 1, False],
                                     [3,   64,  16,  24, False, 'RE', 2, False],
                                     [3,   72,  24,  24, False, 'RE', 1, False],
                                     [5,   72,  24,  40,  True, 'RE', 2, False],
                                     [5,  120,  40,  40,  True, 'RE', 1, False],
                                     [5,  120,  40,  40,  True, 'RE', 1, False],
                                     [3,  240,  40,  80, False, 'HS', 2, False],
                                     [3,  200,  80,  80, False, 'HS', 1, False],
                                     [3,  184,  80,  80, False, 'HS', 1, False],
                                     [3,  184,  80,  80, False, 'HS', 1, False],
                                     [3,  480,  80, 112,  True, 'HS', 1, False],
                                     [3,  672, 112, 112,  True, 'HS', 1, False],
                                     [5,  672, 112, 160,  True, 'HS', 1, True],
                                     [5,  672, 160, 160,  True, 'HS', 2, False],
                                     [5,  960, 160, 160,  True, 'HS', 1, False]]

        self._base_layers.append(gen_init_conv_bn(3, 16, 2))

        for config in self._block_layer_configs:
            self._base_layers.append(gen_block_layer(config))

        self._base_layers.append(nn.Sequential(
            nn.Conv2d(160, 960, 1, bias=False),
            nn.BatchNorm2d(960),
            HardSwish(),
        ))

        self._extra_layers = nn.ModuleList([
            Block(in_channels=960, out_channels=512,
                  hidden_channels=int(1280 * 0.2)),
            Block(in_channels=512, out_channels=256,
                  hidden_channels=int(512 * 0.25)),
            Block(in_channels=256, out_channels=256,
                  hidden_channels=int(256 * 0.5)),
            Block(in_channels=256, out_channels=64,
                  hidden_channels=int(256 * 0.25))
        ])

        self._multibox_layer = MultiBoxLayer(n_classes=self._n_classes)

        self.extras = self._extra_layers
        self.loc = self._multibox_layer.regression_heads
        self.conf = self._multibox_layer.classification_heads

    def forward(self, x):
        confs = []
        locs = []
        hs = []
        # Forward pass through base layers
        for k in range(13):
            x = self._base_layers[k](x)

        # C4 layer
        x, h = self._base_layers[13](x)
        hs.append(h)

        for k in range(14, len(self._base_layers)):
            x = self._base_layers[k](x)

        # Last layer in base layers
        hs.append(x)

        # Forward pass through extra layers
        for layer in self._extra_layers:
            x = layer(x)
            hs.append(x)

        loc_preds, conf_preds = self._multibox_layer(hs)

        if self._phase == 'test':
            pass
        else:
            output = (
                loc_preds,
                conf_preds,
                self.priors,
            )


        return output


def build_ssdlite(phase, n_classes):
    return SSDLite(phase, n_classes)