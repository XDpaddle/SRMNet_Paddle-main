import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddleseg.cvlibs import param_init


class conv_2d(nn.Layer):
    def __init__(self, in_ch, out_ch, kernel, activation='relu', bias=True, padding=1):
        super(conv_2d, self).__init__()
        if activation == 'relu':
            self.conv = nn.Sequential(
                nn.Conv2D(in_ch, out_ch, kernel_size=kernel, bias_attr=bias, padding=padding),
                nn.BatchNorm2D(out_ch),
                nn.ReLU()
            )
        elif activation == 'leakyrelu':
            self.conv = nn.Sequential(
                nn.Conv2D(in_ch, out_ch, kernel_size=kernel, bias_attr=bias, padding=padding),
                nn.BatchNorm2D(out_ch),
                nn.LeakyReLU(negative_slope=0.2)
            )

    def forward(self, x):
        x = self.conv(x)
        return x


class upsample_and_concat(nn.Layer):
    def __init__(self, in_ch, out_ch, Transpose=True):
        super(upsample_and_concat, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if Transpose:
            self.up = nn.Conv2DTranspose(in_ch, out_ch, kernel_size=2, stride=2)
        else:
            self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                    nn.Conv2D(in_ch, out_ch, kernel_size=2, padding=0),
                                    nn.ReLU())
        self.up.apply(self.init_weights)

    def forward(self, x1, x2):
        '''
            conv output shape = (input_shape - Filter_shape + 2 * padding)/stride + 1
        '''
        x1 = self.up(x1)
        diffY = x2.shape[2] - x1.shape[2]
        diffX = x2.shape[3] - x1.shape[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX//2, diffY // 2, diffY - diffY//2])
        x = paddle.concat([x2, x1], axis=1)
        return x

    @staticmethod
    def init_weights(m):
        if type(m) == nn.Conv2D:
            param_init.xavier_normal_init(m.weight)
            param_init.constant_init(m.bias, 0)


class Unet(nn.Layer):
    def __init__(self):
        super(Unet, self).__init__()
        self.conv1_0 = conv_2d(2, 32, 3, activation='leakyrelu')
        self.conv1_1 = conv_2d(32, 32, 3, activation='leakyrelu')
        self.conv1_2 = conv_2d(32, 32, 3, activation='leakyrelu')
        self.conv1_3 = conv_2d(32, 32, 3, activation='leakyrelu')

        self.conv2_0 = conv_2d(32, 64, 3, activation='leakyrelu')
        self.conv2_1 = conv_2d(64, 64, 3, activation='leakyrelu')
        self.conv2_2 = conv_2d(64, 64, 3, activation='leakyrelu')
        self.conv2_3 = conv_2d(64, 64, 3, activation='leakyrelu')

        self.conv3_0 = conv_2d(64, 128, 3, activation='leakyrelu')
        self.conv3_1 = conv_2d(128, 128, 3, activation='leakyrelu')
        self.conv3_2 = conv_2d(128, 128, 3, activation='leakyrelu')
        self.conv3_3 = conv_2d(128, 128, 3, activation='leakyrelu')

        self.conv4_0 = conv_2d(128, 256, 3, activation='leakyrelu')
        self.conv4_1 = conv_2d(256, 256, 3, activation='leakyrelu')
        self.conv4_2 = conv_2d(256, 256, 3, activation='leakyrelu')
        self.conv4_3 = conv_2d(256, 256, 3, activation='leakyrelu')

        self.conv5_0 = conv_2d(256, 512, 3, activation='leakyrelu')
        self.conv5_1 = conv_2d(512, 512, 3, activation='leakyrelu')
        self.conv5_2 = conv_2d(512, 512, 3, activation='leakyrelu')
        self.conv5_3 = conv_2d(512, 512, 3, activation='leakyrelu')

        self.up6 = upsample_and_concat(512, 256)

        self.conv6_0 = conv_2d(512, 256, 3, activation='leakyrelu')
        self.conv6_1 = conv_2d(256, 256, 3, activation='leakyrelu')
        self.conv6_2 = conv_2d(256, 256, 3, activation='leakyrelu')

        self.up7 = upsample_and_concat(256, 128)

        self.conv7_0 = conv_2d(256, 128, 3, activation='leakyrelu')
        self.conv7_1 = conv_2d(128, 128, 3, activation='leakyrelu')
        self.conv7_2 = conv_2d(128, 128, 3, activation='leakyrelu')

        self.up8 = upsample_and_concat(128, 64)

        self.conv8_0 = conv_2d(128, 64, 3, activation='leakyrelu')
        self.conv8_1 = conv_2d(64, 64, 3, activation='leakyrelu')
        self.conv8_2 = conv_2d(64, 64, 3, activation='leakyrelu')

        self.up9 = upsample_and_concat(64, 32)

        self.conv9_0 = conv_2d(64, 32, 3, activation='leakyrelu')
        self.conv9_1 = conv_2d(32, 32, 3, activation='leakyrelu')
        self.conv9_2 = conv_2d(32, 32, 3, activation='leakyrelu')

        self.conv10_0 = conv_2d(32, 1, 1, activation='leakyrelu', padding=0)
        
    def forward(self, x):
        conv1 = self.conv1_0(x)
        conv1 = self.conv1_1(conv1)
        conv1 = self.conv1_2(conv1)
        conv1 = self.conv1_3(conv1)
        pool1 = F.max_pool2d(conv1, [2, 2], padding=1)  # [32, 32, 15, 15]

        conv2 = self.conv2_0(pool1)
        conv2 = self.conv2_1(conv2)
        conv2 = self.conv2_2(conv2)
        conv2 = self.conv2_3(conv2)
        pool2 = F.max_pool2d(conv2, [2, 2], padding=1)  # [32, 64, 8, 8]

        conv3 = self.conv3_0(pool2)
        conv3 = self.conv3_1(conv3)
        conv3 = self.conv3_2(conv3)
        conv3 = self.conv3_3(conv3)
        pool3 = F.max_pool2d(conv3, [2, 2], padding=1)  # [32, 64, 5, 5]

        conv4 = self.conv4_0(pool3)
        conv4 = self.conv4_1(conv4)
        conv4 = self.conv4_2(conv4)
        conv4 = self.conv4_3(conv4)
        pool4 = F.max_pool2d(conv4, [2, 2], padding=1)  # [32, 64, 3, 3]

        conv5 = self.conv5_0(pool4)
        conv5 = self.conv5_1(conv5)
        conv5 = self.conv5_2(conv5)
        conv5 = self.conv5_3(conv5)

        up6 = self.up6(conv5, conv4)
        conv6 = self.conv6_0(up6)
        conv6 = self.conv6_1(conv6)
        conv6 = self.conv6_2(conv6)

        up7 = self.up7(conv6, conv3)
        conv7 = self.conv7_0(up7)
        conv7 = self.conv7_1(conv7)
        conv7 = self.conv7_2(conv7)

        up8 = self.up8(conv7, conv2)
        conv8 = self.conv8_0(up8)
        conv8 = self.conv8_1(conv8)
        conv8 = self.conv8_2(conv8)

        up9 = self.up9(conv8, conv1)
        conv9 = self.conv9_0(up9)
        conv9 = self.conv9_1(conv9)
        conv9 = self.conv9_2(conv9)

        conv10 = self.conv10_0(conv9)

        return conv10


class All_Unet(nn.Layer):
    def __init__(self):
        super(All_Unet, self).__init__()
        self.unet1 = Unet()
        self.unet2 = Unet()
        self.unet3 = Unet()
        self.unet4 = Unet()
        self.unet5 = Unet()

    def forward(self, pool1, pool2, pool3):
        unet1 = self.unet1(pool1)
        unet2 = self.unet2(pool2)
        unet3 = self.unet3(pool3)
        # unet4 = self.unet4(pool4)
        # unet5 = self.unet5(pool5)

        return unet1, unet2, unet3


class Encoder(nn.Layer):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = conv_2d(1, 32, 3, activation='relu')
        self.conv2 = conv_2d(32, 32, 3, activation='leakyrelu')
        self.conv3 = conv_2d(32, 32, 3, activation='leakyrelu')
        self.conv4 = conv_2d(32, 32, 3, activation='leakyrelu')
        self.conv5 = conv_2d(32, 1, 1, activation='leakyrelu', padding=0)
        self.cal = channel_attention_layer(32, 2, 32)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv4 = self.cal(conv4)
        output = self.conv5(conv4)
        return output


class Decoder(nn.Layer):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv1 = conv_2d(5, 7, 3, activation='leakyrelu', padding=1)
        self.conv2 = conv_2d(5, 7, 5, activation='leakyrelu', padding=2)
        self.conv3 = conv_2d(5, 7, 7, activation='leakyrelu', padding=3)
        self.conv4 = conv_2d(7, 1, 3, activation='leakyrelu', padding=1)
        self.ksl = kernel_select_layer(7, 4, 7)

    def forward(self, x):
        sk_conv1 = self.conv1(x)
        sk_conv2 = self.conv2(x)
        sk_conv3 = self.conv3(x)
        sk_out = self.ksl(sk_conv1, sk_conv2, sk_conv3)
        output = self.conv4(sk_out)
        return output


class channel_attention_layer(nn.Layer):
    def __init__(self, in_dim, middle, out_dim):
        super(channel_attention_layer, self).__init__()
        self.out_dim = out_dim
        self.linear1 = nn.Linear(in_dim, middle, bias_attr=True)
        self.linear2 = nn.Linear(middle, out_dim, bias_attr=True)

    def forward(self, x):
        squeeze = F.adaptive_avg_pool2d(x, (1, 1)).squeeze()  # [32, 32] GAP
        excitation = self.linear1(squeeze)
        excitation = paddle.relu(excitation)
        excitation = self.linear2(excitation)
        excitation = paddle.sigmoid(excitation)
        excitation = paddle.reshape(excitation, [-1, self.out_dim, 1, 1])  # [32, 32, 1, 1]
        scale = x * excitation
        return scale


class kernel_select_layer(nn.Layer):
    def __init__(self, in_dim, middle, out_dim):
        super(kernel_select_layer, self).__init__()
        self.out_dim = out_dim
        self.linear0 = nn.Linear(in_dim, middle, bias_attr=True)
        self.linear1 = nn.Linear(middle, out_dim, bias_attr=True)
        self.linear2 = nn.Linear(middle, out_dim, bias_attr=True)
        self.linear3 = nn.Linear(middle, out_dim, bias_attr=True)

    def forward(self, sk_conv1, sk_conv2, sk_conv3):
        sum_u = sk_conv1 + sk_conv2 + sk_conv3
        squeeze = F.adaptive_avg_pool2d(sum_u, (1, 1)).squeeze()
        # squeeze = torch.reshape(squeeze, [-1, 1, 1, self.out_dim])
        z = self.linear0(squeeze)
        z = F.relu(z)
        a1 = paddle.reshape(self.linear1(z),[-1, 1, self.out_dim])
        a2 = paddle.reshape(self.linear2(z),[-1, 1, self.out_dim])
        a3 = paddle.reshape(self.linear3(z),[-1, 1, self.out_dim])

        before_softmax = paddle.concat([a1, a2, a3], 1)
        after_softmax = paddle.softmax(before_softmax, dim=1)
        a1 = after_softmax[:, 0, :]
        a1 = paddle.reshape(a1, [-1, self.out_dim, 1, 1])
        a2 = after_softmax[:, 1, :]
        a2 = paddle.reshape(a2,  [-1, self.out_dim, 1, 1])
        a3 = after_softmax[:, 2, :]
        a3 = paddle.reshape(a3,  [-1, self.out_dim, 1, 1])

        select_1 = sk_conv1 * a1
        select_2 = sk_conv2 * a2
        select_3 = sk_conv3 * a3

        out = select_1 + select_2 + select_3

        return out


def avg_pool(feature_map):
    pool1 = F.avg_pool2d(feature_map, kernel_size=1, stride=1, padding=0)
    pool2 = F.avg_pool2d(feature_map, kernel_size=2, stride=2, padding=0)
    pool3 = F.avg_pool2d(feature_map, kernel_size=4, stride=4, padding=0)
    # pool4 = F.avg_pool2d(feature_map, kernel_size=8, stride=8, padding=0)
    # pool5 = F.avg_pool2d(feature_map, kernel_size=16, stride=16, padding=0)

    return pool1, pool2, pool3


def resize_all_image(unet1, unet2, unet3):
    resize1 = F.interpolate(unet1, scale_factor=1, mode='bilinear', align_corners=False)
    resize2 = F.interpolate(unet2, scale_factor=2, mode='bilinear', align_corners=False)
    resize3 = F.interpolate(unet3, scale_factor=4, mode='bilinear', align_corners=False)
    # resize4 = F.interpolate(unet4, scale_factor=8, mode='bilinear', align_corners=False)
    # resize5 = F.interpolate(unet5, scale_factor=16, mode='bilinear', align_corners=False)

    return resize1, resize2, resize3


class PRIDNet(nn.Layer):
    def __init__(self):
        super(PRIDNet, self).__init__()
        self.encoder = Encoder()
        self.all_unet = All_Unet()
        self.decoder = Decoder()

    def forward(self, x):
        feature_map = self.encoder(x)
        feature_map_2 = paddle.concat([x, feature_map], 1)  # [32, 2, 28, 28]
        pool1, pool2, pool3 = avg_pool(feature_map_2)  # [32, 2, 28, 28] [32, 2, 14, 14] [32, 2, 7, 7]
        unet1, unet2, unet3 = self.all_unet(pool1, pool2, pool3)
        resize1, resize2, resize3 = resize_all_image(unet1, unet2, unet3)
        fea_cat = paddle.concat([feature_map_2, resize1, resize2, resize3], 1)
        out_image = self.decoder(fea_cat)

        return out_image