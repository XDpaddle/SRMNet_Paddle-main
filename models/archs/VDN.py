from . import DnCNN, UNet
import paddle.nn as nn
from paddleseg.cvlibs import param_init
"""
Class for VDN network 
(combination of U-Net and DnCNN)
"""


class VDN_NET(nn.Layer):
    def __init__(self, in_channels, depth_snet=5):
        super(VDN_NET, self).__init__()
        d_net = UNet.UNet(in_channels=in_channels, out_channels=in_channels*2)
        s_net = DnCNN.DnCNN(DnCNN.make_Layers(depth_snet, out_channels=in_channels*2), image_channels=in_channels)
        self.DNet = self.init_kaiming(d_net)
        self.SNet = self.init_kaiming(s_net)

    def init_kaiming(self, net):
        for layer in net.named_children():
            if isinstance(layer, nn.Conv2D):
                param_init.kaiming_normal_init(layer.weight)
            elif isinstance(layer, nn.BatchNorm2D):
                param_init.constant_init(layer.weight, 1)
                param_init.constant_init(layer.bias, 0)
        return net

    def forward(self, x):  # train mode
        Z = self.DNet(x)
        sigma = self.SNet(x)
        return Z, sigma