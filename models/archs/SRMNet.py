import paddle
import paddle.nn as nn
from thop import profile


##---------- Basic Layers ----------
def conv3x3(in_chn, out_chn, bias=True):
    layer = nn.Conv2D(in_chn, out_chn, kernel_size=3, stride=1, padding=1, bias_attr=bias)
    return layer

def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2D(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias_attr=bias, stride=stride)

def bili_resize(factor):
    return nn.Upsample(scale_factor=factor, mode='bilinear', align_corners=False)

##---------- Basic Blocks ----------
class UNetConvBlock(nn.Layer):
    def __init__(self, in_size, out_size, downsample):
        super(UNetConvBlock, self).__init__()
        self.downsample = downsample
        self.block = SK_RDB(in_channels=in_size, growth_rate=out_size, num_layers=3)
        if downsample:
            self.downsample = PS_down(out_size, out_size, downscale=2)

    def forward(self, x):
        out = self.block(x)
        if self.downsample:
            out_down = self.downsample(out)
            return out_down, out
        else:
            return out

class UNetUpBlock(nn.Layer):
    def __init__(self, in_size, out_size):
        super(UNetUpBlock, self).__init__()
        # self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, bias=True)
        self.up = PS_up(in_size, out_size, upscale=2)
        self.conv_block = UNetConvBlock(in_size, out_size, False)

    def forward(self, x, bridge):
        up = self.up(x)
        out = paddle.concat([up, bridge], axis=1)
        out = self.conv_block(out)
        return out

##---------- Resizing Modules (Pixel(Un)Shuffle) ----------
class PS_down(nn.Layer):
    def __init__(self, in_size, out_size, downscale):
        super(PS_down, self).__init__()
        self.UnPS = nn.PixelUnshuffle(downscale)
        self.conv1 = nn.Conv2D((downscale**2) * in_size, out_size, 1, 1, 0)

    def forward(self, x):
        x = self.UnPS(x)  # h/2, w/2, 4*c
        x = self.conv1(x)
        return x

class PS_up(nn.Layer):
    def __init__(self, in_size, out_size, upscale):
        super(PS_up, self).__init__()

        self.PS = nn.PixelShuffle(upscale)
        self.conv1 = nn.Conv2D(in_size//(upscale**2), out_size, 1, 1, 0)

    def forward(self, x):
        x = self.PS(x)  # h/2, w/2, 4*c
        x = self.conv1(x)
        return x

##---------- Selective Kernel Feature Fusion (SKFF) ----------
class SKFF(nn.Layer):
    def __init__(self, in_channels, height=3, reduction=8, bias=False):
        super(SKFF, self).__init__()

        self.height = height
        d = max(int(in_channels / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.conv_du = nn.Sequential(nn.Conv2D(in_channels, d, 1, padding=0, bias_attr=bias), nn.PReLU())

        self.fcs = nn.LayerList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2D(d, in_channels, kernel_size=1, stride=1, bias_attr=bias))

        self.softmax = nn.Softmax(axis=1)

    def forward(self, inp_feats):
        batch_size, n_feats, H, W = inp_feats[1].shape

        inp_feats = paddle.concat(inp_feats, axis=1)
        inp_feats = paddle.reshape(inp_feats,[batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3]])

        feats_U = paddle.sum(inp_feats, axis=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = paddle.concat(attention_vectors, axis=1)
        attention_vectors = paddle.reshape(attention_vectors,[batch_size, self.height, n_feats, 1, 1])

        attention_vectors = self.softmax(attention_vectors)
        feats_V = paddle.sum(inp_feats * attention_vectors, axis=1)

        return feats_V

##---------- Dense Block ----------
class DenseLayer(nn.Layer):
    def __init__(self, in_channels, out_channels, I):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2D(in_channels, out_channels, kernel_size=3, padding=3 // 2)
        self.relu = nn.ReLU()
        self.sk = SKFF(out_channels, height=2, reduction=8, bias=False)

    def forward(self, x):
        x1 = self.relu(self.conv(x))
        # output = torch.cat([x, x1], 1) # -> RDB
        output = self.sk((x, x1))
        return output

##---------- Selective Kernel Residual Dense Block (SK-RDB) ----------
class SK_RDB(nn.Layer):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(SK_RDB, self).__init__()
        self.identity = nn.Conv2D(in_channels, growth_rate, 1, 1, 0)
        self.layers = nn.Sequential(
            *[DenseLayer(in_channels, in_channels, I=i) for i in range(num_layers)]
        )
        self.lff = nn.Conv2D(in_channels, growth_rate, kernel_size=1)

    def forward(self, x):
        res = self.identity(x)
        x = self.layers(x)
        x = self.lff(x)
        return res + x

##---------- testNet ----------
class SRMNet(nn.Layer):
    def __init__(self, in_chn=3, wf=96, depth=4):
        super(SRMNet, self).__init__()
        self.depth = depth
        self.down_path = nn.LayerList()
        self.bili_down = bili_resize(0.5)
        self.conv_01 = nn.Conv2D(in_chn, wf, 3, 1, 1)

        # encoder of UNet
        prev_channels = 0
        for i in range(depth):  # 0,1,2,3
            downsample = True if (i + 1) < depth else False
            self.down_path.append(UNetConvBlock(prev_channels + wf, (2 ** i) * wf, downsample))
            prev_channels = (2 ** i) * wf

        # decoder of UNet
        self.up_path = nn.LayerList()
        self.skip_conv = nn.LayerList()
        self.conv_up = nn.LayerList()
        self.bottom_conv = nn.Conv2D(prev_channels, wf, 3, 1, 1)
        self.bottom_up = bili_resize(2 ** (depth-1))

        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock(prev_channels, (2 ** i) * wf))
            self.skip_conv.append(nn.Conv2D((2 ** i) * wf, (2 ** i) * wf, 3, 1, 1))
            self.conv_up.append(nn.Sequential(*[nn.Conv2D((2 ** i) * wf, wf, 3, 1, 1), bili_resize(2 ** i)]))
            prev_channels = (2 ** i) * wf

        self.final_ff = SKFF(in_channels=wf, height=depth)
        self.last = conv3x3(prev_channels, in_chn, bias=True)

    def forward(self, x):
        img = x
        scale_img = img

        ##### shallow conv #####
        x1 = self.conv_01(img)
        encs = []
        ######## UNet ########
        # Down-path (Encoder)
        for i, down in enumerate(self.down_path):
            if i == 0:
                x1, x1_up = down(x1)
                encs.append(x1_up)
            elif (i + 1) < self.depth:
                scale_img = self.bili_down(scale_img)
                left_bar = self.conv_01(scale_img)
                x1 = paddle.concat([x1, left_bar], axis=1)
                x1, x1_up = down(x1)
                encs.append(x1_up)
            else:
                scale_img = self.bili_down(scale_img)
                left_bar = self.conv_01(scale_img)
                x1 = paddle.concat([x1, left_bar], axis=1)
                x1 = down(x1)

        # Up-path (Decoder)
        ms_result = [self.bottom_up(self.bottom_conv(x1))]
        for i, up in enumerate(self.up_path):
            x1 = up(x1, self.skip_conv[i](encs[-i - 1]))
            ms_result.append(self.conv_up[i](x1))
        # Multi-scale selective feature fusion
        msff_result = self.final_ff(ms_result)

        ##### Reconstruct #####
        out_1 = self.last(msff_result) + img

        return out_1