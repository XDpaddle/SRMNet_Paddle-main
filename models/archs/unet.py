import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class ConvBlock(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dropout=False,
                 norm='batch',
                 residual=True,
                 activation='leakyrelu',
                 transpose=False):
        super(ConvBlock, self).__init__()
        self.dropout = dropout
        self.residual = residual
        self.activation = activation
        self.transpose = transpose

        if self.dropout:
            self.dropout1 = nn.Dropout2D(p=0.05)
            self.dropout2 = nn.Dropout2D(p=0.05)

        self.norm1 = None
        self.norm2 = None
        if norm == 'batch':
            self.norm1 = nn.BatchNorm2D(out_channels)
            self.norm2 = nn.BatchNorm2D(out_channels)
        elif norm == 'instance':
            self.norm1 = nn.InstanceNorm2D(out_channels)
            self.norm2 = nn.InstanceNorm2D(out_channels)
        elif norm == 'mixed':
            self.norm1 = nn.BatchNorm2D(out_channels)
            self.norm2 = nn.InstanceNorm2D(out_channels)

        if self.transpose:
            self.conv1 = nn.ConvTranspose2D(in_channels,
                                            out_channels,
                                            kernel_size=3,
                                            padding=1)
            self.conv2 = nn.ConvTranspose2D(out_channels,
                                            out_channels,
                                            kernel_size=3,
                                            padding=1)
        else:
            self.conv1 = nn.Conv2D(in_channels,
                                   out_channels,
                                   kernel_size=3,
                                   padding=1)
            self.conv2 = nn.Conv2D(out_channels,
                                   out_channels,
                                   kernel_size=3,
                                   padding=1)

        if self.activation == 'relu':
            self.actfun1 = nn.ReLU()
            self.actfun2 = nn.ReLU()
        elif self.activation == 'leakyrelu':
            self.actfun1 = nn.LeakyReLU()
            self.actfun2 = nn.LeakyReLU()
        elif self.activation == 'elu':
            self.actfun1 = nn.ELU()
            self.actfun2 = nn.ELU()
        elif self.activation == 'selu':
            self.actfun1 = nn.SELU()
            self.actfun2 = nn.SELU()

    def forward(self, x):
        ox = x

        x = self.conv1(x)

        if self.dropout:
            x = self.dropout1(x)

        if self.norm1:
            x = self.norm1(x)

        x = self.actfun1(x)

        x = self.conv2(x)

        if self.dropout:
            x = self.dropout2(x)

        if self.norm2:
            x = self.norm2(x)

        if self.residual:
            x[:, 0:min(ox.shape[1], x.shape[1]
                       ), :, :] += ox[:, 0:min(ox.shape[1], x.shape[1]), :, :]

        x = self.actfun2(x)

        # print("shapes: x:%s ox:%s " % (x.shape,ox.shape))

        return x
    

class Unet(nn.Layer):
    def __init__(self,
                 n_channel_in=1,
                 n_channel_out=1,
                 residual=False,
                 down='conv',
                 up='tconv',
                 activation='selu',
                 post_processing=False):
        super(Unet, self).__init__()

        self.residual = residual
        self.post_process = post_processing

        if down == 'maxpool':
            self.down1 = nn.MaxPool2D(kernel_size=2)
            self.down2 = nn.MaxPool2D(kernel_size=2)
            self.down3 = nn.MaxPool2D(kernel_size=2)
            self.down4 = nn.MaxPool2D(kernel_size=2)
        elif down == 'avgpool':
            self.down1 = nn.AvgPool2D(kernel_size=2)
            self.down2 = nn.AvgPool2D(kernel_size=2)
            self.down3 = nn.AvgPool2D(kernel_size=2)
            self.down4 = nn.AvgPool2D(kernel_size=2)
        elif down == 'conv':
            self.down1 = nn.Conv2D(32, 32, kernel_size=2, stride=2, groups=32)
            self.down2 = nn.Conv2D(64, 64, kernel_size=2, stride=2, groups=64)
            self.down3 = nn.Conv2D(128,
                                   128,
                                   kernel_size=2,
                                   stride=2,
                                   groups=128)
            self.down4 = nn.Conv2D(256,
                                   256,
                                   kernel_size=2,
                                   stride=2,
                                   groups=256)

            x = 0.01 * self.down1.weight + 0.25
            self.down1.weight = paddle.create_parameter(shape=x.shape,
                                                        dtype='float32',
                                                        default_initializer=paddle.nn.initializer.Assign(x))
            x = 0.01 * self.down2.weight + 0.25
            self.down2.weight = paddle.create_parameter(shape=x.shape,
                                                        dtype='float32',
                                                        default_initializer=paddle.nn.initializer.Assign(x))
            #self.down3.weight = 0.01 * self.down3.weight + 0.25
            x = 0.01 * self.down3.weight + 0.25
            self.down3.weight = paddle.create_parameter(shape=x.shape,
                                                        dtype='float32',
                                                        default_initializer=paddle.nn.initializer.Assign(x))
            #self.down4.weight = 0.01 * self.down4.weight + 0.25
            x = 0.01 * self.down4.weight + 0.25
            self.down4.weight = paddle.create_parameter(shape=x.shape,
                                                        dtype='float32',
                                                        default_initializer=paddle.nn.initializer.Assign(x))

            #self.down1.bias = 0.01 * self.down1.bias + 0
            x = 0.01 * self.down1.bias + 0
            self.down1.bias = paddle.create_parameter(shape=x.shape,
                                                        dtype='float32',
                                                        default_initializer=paddle.nn.initializer.Assign(x))
            #self.down2.bias = 0.01 * self.down2.bias + 0
            x = 0.01 * self.down2.bias + 0
            self.down2.bias = paddle.create_parameter(shape=x.shape,
                                                        dtype='float32',
                                                        default_initializer=paddle.nn.initializer.Assign(x))
            #self.down3.bias = 0.01 * self.down3.bias + 0
            x = 0.01 * self.down3.bias + 0
            self.down3.bias = paddle.create_parameter(shape=x.shape,
                                                        dtype='float32',
                                                        default_initializer=paddle.nn.initializer.Assign(x))
            #self.down4.bias = 0.01 * self.down4.bias + 0
            x = 0.01 * self.down4.bias + 0
            self.down4.bias = paddle.create_parameter(shape=x.shape,
                                                        dtype='float32',
                                                        default_initializer=paddle.nn.initializer.Assign(x))

        if up == 'bilinear' or up == 'nearest':
            self.up1 = lambda x: nn.functional.interpolate(
                x, mode=up, scale_factor=2)
            self.up2 = lambda x: nn.functional.interpolate(
                x, mode=up, scale_factor=2)
            self.up3 = lambda x: nn.functional.interpolate(
                x, mode=up, scale_factor=2)
            self.up4 = lambda x: nn.functional.interpolate(
                x, mode=up, scale_factor=2)
        elif up == 'tconv':
            self.up1 = nn.Conv2DTranspose(256,
                                          256,
                                          kernel_size=2,
                                          stride=2,
                                          groups=256)
            self.up2 = nn.Conv2DTranspose(128,
                                          128,
                                          kernel_size=2,
                                          stride=2,
                                          groups=128)
            self.up3 = nn.Conv2DTranspose(64,
                                          64,
                                          kernel_size=2,
                                          stride=2,
                                          groups=64)
            self.up4 = nn.Conv2DTranspose(32,
                                          32,
                                          kernel_size=2,
                                          stride=2,
                                          groups=32)

            #self.up1.weight = 0.01 * self.up1.weight + 0.25
            x = 0.01 * self.up1.weight + 0.25
            self.up1.weight = paddle.create_parameter(shape=x.shape,
                                                        dtype='float32',
                                                        default_initializer=paddle.nn.initializer.Assign(x))
            #self.up2.weight = 0.01 * self.up2.weight + 0.25
            x = 0.01 * self.up2.weight + 0.25
            self.up2.weight = paddle.create_parameter(shape=x.shape,
                                                        dtype='float32',
                                                        default_initializer=paddle.nn.initializer.Assign(x))
            #self.up3.weight= 0.01 * self.up3.weight + 0.25
            x = 0.01 * self.up3.weight + 0.25
            self.up3.weight = paddle.create_parameter(shape=x.shape,
                                                        dtype='float32',
                                                        default_initializer=paddle.nn.initializer.Assign(x))
            #self.up4.weight = 0.01 * self.up4.weight + 0.25
            x = 0.01 * self.up4.weight + 0.25
            self.up4.weight = paddle.create_parameter(shape=x.shape,
                                                        dtype='float32',
                                                        default_initializer=paddle.nn.initializer.Assign(x))

            #self.up1.bias = 0.01 * self.up1.bias + 0
            x = 0.01 * self.up1.bias + 0
            self.up1.bias = paddle.create_parameter(shape=x.shape,
                                                        dtype='float32',
                                                        default_initializer=paddle.nn.initializer.Assign(x))
            #self.up2.bias = 0.01 * self.up2.bias + 0
            x = 0.01 * self.up2.bias + 0
            self.up2.bias = paddle.create_parameter(shape=x.shape,
                                                        dtype='float32',
                                                        default_initializer=paddle.nn.initializer.Assign(x))
            #self.up3.bias = 0.01 * self.up3.bias + 0
            x = 0.01 * self.up3.bias + 0
            self.up3.bias = paddle.create_parameter(shape=x.shape,
                                                        dtype='float32',
                                                        default_initializer=paddle.nn.initializer.Assign(x))
            #self.up4.bias = 0.01 * self.up4.bias + 0
            x = 0.01 * self.up4.bias + 0
            self.up4.bias = paddle.create_parameter(shape=x.shape,
                                                        dtype='float32',
                                                        default_initializer=paddle.nn.initializer.Assign(x))

        self.conv1 = ConvBlock(n_channel_in, 32, residual, activation)
        self.conv2 = ConvBlock(32, 64, residual, activation)
        self.conv3 = ConvBlock(64, 128, residual, activation)
        self.conv4 = ConvBlock(128, 256, residual, activation)
        self.conv5 = ConvBlock(256, 256, residual, activation)

        self.conv6 = ConvBlock(2 * 256, 128, residual, activation)
        self.conv7 = ConvBlock(2 * 128, 64, residual, activation)
        self.conv8 = ConvBlock(2 * 64, 32, residual, activation)
        self.conv9 = ConvBlock(2 * 32, 64, residual, activation)

        if self.residual:
            self.convres = ConvBlock(n_channel_in, n_channel_out, residual,
                                     activation)
        if self.post_process:
            self.conv1x1 = nn.Sequential(nn.Conv2D(64, 64, 1,
                                                   1), nn.LeakyReLU(),
                                         nn.ConvD(64, 64, 1,
                                                   1), nn.LeakyReLU(),
                                         nn.Conv2D(64, 64, 1, 1),
                                         nn.LeakyReLU(),
                                         nn.Conv2D(64, n_channel_out, 1, 1),
                                         nn.ReLU())

    def forward(self, x):
        c0 = x
        c1 = self.conv1(x)
        x = self.down1(c1)
        c2 = self.conv2(x)
        x = self.down2(c2)
        c3 = self.conv3(x)
        x = self.down3(c3)
        c4 = self.conv4(x)
        x = self.down4(c4)
        x = self.conv5(x)
        x = self.up1(x)
        # print("shapes: c0:%sx:%s c4:%s " % (c0.shape,x.shape,c4.shape))
        x = paddle.concat([x, c4], 1)  # x[:,0:128]*x[:,128:256],
        x = self.conv6(x)
        x = self.up2(x)
        x = paddle.concat([x, c3], 1)  # x[:,0:64]*x[:,64:128],
        x = self.conv7(x)
        x = self.up3(x)
        x =paddle.concat([x, c2], 1)  # x[:,0:32]*x[:,32:64],
        x = self.conv8(x)
        x = self.up4(x)
        x = paddle.concat([x, c1], 1)  # x[:,0:16]*x[:,16:32],
        x = self.conv9(x)
        if self.residual:
            x = paddle.add(x, self.convres(c0))
        if self.post_process:
            x = self.conv1x1(x)

        return x