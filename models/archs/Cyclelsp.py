

import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

##########################################################################

def conv(in_channels, out_channels, kernel_size, bias=True, padding = 1, stride = 1):
    return nn.Conv2D(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias_attr=bias, stride = stride)



##########################################################################

## Channel Attention (CA) Layer
class CALayer(nn.Layer):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2D(channel, channel // reduction, 1, padding=0, bias_attr=True),
                nn.ReLU(),
                nn.Conv2D(channel // reduction, channel, 1, padding=0, bias_attr=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y
##########################################################################

class BasicConv(nn.Layer):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=False, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2D(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias_attr=bias)
        self.bn = nn.BatchNorm2D(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ChannelPool(nn.Layer):
    def forward(self, x):
        return paddle.concat( (paddle.max(x,1)[0].unsqueeze(1), paddle.mean(x,1).unsqueeze(1)), axis=1 )

class spatial_attn_layer(nn.Layer):
    def __init__(self, kernel_size=3):
        super(spatial_attn_layer, self).__init__()
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        # import pdb;pdb.set_trace()
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = nn.Sigmoid(x_out) # broadcasting
        return x * scale

##########################################################################


## Dual Attention Block (DAB)
class DAB(nn.Layer):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True)):

        super(DAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2D(n_feat))
            if i == 0: modules_body.append(act)
        
        self.SA = spatial_attn_layer()            ## Spatial Attention
        self.CA = CALayer(n_feat, reduction)     ## Channel Attention
        self.body = nn.Sequential(*modules_body)
        self.conv1x1 = nn.Conv2D(n_feat*2, n_feat, kernel_size=1)


    def forward(self, x):
        res = self.body(x)
        sa_branch = self.SA(res)
        ca_branch = self.CA(res)
        res = paddle.concat([sa_branch, ca_branch], axis=1)
        res = self.conv1x1(res)
        res += x
        return res

##########################################################################


## Recursive Residual Group (RRG)
class RRG(nn.Layer):
    def __init__(self, conv, n_feat, kernel_size, reduction, act,  num_dab):
        super(RRG, self).__init__()
        modules_body = []
        modules_body = [
            DAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=act) \
            for _ in range(num_dab)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

##########################################################################


class DenoiseNet(nn.Layer):
    def __init__(self, conv=conv):
        super(DenoiseNet, self).__init__()
        num_rrg = 4
        num_dab = 8
        n_feats = 64
        kernel_size = 3
        reduction = 16 
        inp_chans = 8   # 4 RGGB channels, and 4 Variance maps
        act =nn.PReLU(n_feats)
        
        modules_head = [conv(inp_chans, n_feats, kernel_size = kernel_size, stride = 1)]

        modules_body = [
            RRG(
                conv, n_feats, kernel_size, reduction, act=act, num_dab=num_dab) \
            for _ in range(num_rrg)]

        modules_body.append(conv(n_feats, n_feats, kernel_size))
        modules_body.append(act)

        modules_tail = [conv(n_feats, inp_chans//2, kernel_size)]


        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)


    def forward(self, noisy_img, variance):
        x = paddle.concat([noisy_img, variance], axis=1)
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        x = noisy_img + x
        return x