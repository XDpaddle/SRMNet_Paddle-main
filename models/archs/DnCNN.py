import paddle.nn as nn
from paddleseg.cvlibs import param_init as init

"""
This is the class for the S-Net
"""


class DnCNN(nn.Layer):
    def __init__(self, last_layers, padding=1, image_channels=1, n_channels=64, kernel_size=3, init_weights=True):
        super(DnCNN, self).__init__()
        self.dncnn1 = nn.Sequential(
            nn.Conv2D(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, stride =1,
                      bias_attr=True),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.dncnn2 = last_layers
        if init_weights:
            self.init_weights()

    def forward(self, x):
        #y = x
        x = self.dncnn1(x)
        x = self.dncnn2(x)
        # x = y-x
        #print(x.shape)
        return x

    def init_weights(self):
        for m in self.named_children():
            if isinstance(m, nn.Conv2D):
                init.normal_init(m.weight)
                if m.bias is not None:
                    init.constant_init(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2D):
                init.constant_init(m.weight, 1)
                init.constant_init(m.bias, 0)


def make_Layers(depth, n_channels=64, out_channels=2, kernel_size=3, padding=1):
    layers = []
    for x in range(depth - 2):
        layers += [nn.Conv2D(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, stride = 1, bias_attr=True),
                   nn.LeakyReLU(negative_slope=0.2)]
    layers += [nn.Conv2D(in_channels=n_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride = 1,
                         bias_attr=True)]
    return nn.Sequential(*layers)