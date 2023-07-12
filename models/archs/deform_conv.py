# torchvisison >= 0.9.0
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddleseg.cvlibs import param_init
from paddle.vision.ops import deform_conv2d
import math
import logging
logger = logging.getLogger('base')

class ModulatedDeformableConv2d(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 bias=True,
                 deformable_groups=1,
                 extra_offset_mask=True, 
                 offset_in_channel=32
                 ):
        super(ModulatedDeformableConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride 
        self.padding = padding 
        self.dilation = dilation
        self.groups = groups
        self.extra_offset_mask = extra_offset_mask

        self.conv_offset_mask = nn.Conv2D(offset_in_channel,
                                     deformable_groups * 3 * kernel_size * kernel_size,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=self.padding,
                                     bias_attr=True)

        self.init_offset()

        #self.weight = nn.Parameter(paddle.to_tensor([out_channels, in_channels // groups, kernel_size, kernel_size]))
        x=paddle.to_tensor([out_channels, in_channels // groups, kernel_size, kernel_size])       # paddle
        self.weight = paddle.create_parameter(shape=x.shape,
                                            dtype='float32',
                                            default_initializer=paddle.nn.initializer.Assign(x))
        if bias:
            #self.bias = nn.Parameter(paddle.to_tensor([out_channels]))
            x=paddle.to_tensor([out_channels])       # paddle
            self.bias = paddle.create_parameter(shape=x.shape,
                                                dtype='float32',
                                                default_initializer=paddle.nn.initializer.Assign(x))
        else:
            self.bias = None
        self.init_weights()

    def init_weights(self):
        n = self.in_channels * self.kernel_size * self.kernel_size
        stdv = 1. / math.sqrt(n)
        param_init.uniform_init(self.weight, low=-stdv, high=stdv)
        if self.bias is not None:
            param_init.constant_init(self.bias,value=0)
    
    def init_offset(self):
        param_init.constant_init(self.conv_offset_mask.weight, value=0.)
        param_init.constant_init(self.conv_offset_mask.bias, value=0.)


    def forward(self, x):

        if self.extra_offset_mask:
            # x = [input, features]
            offset_mask = self.conv_offset_mask(x[1])
            x = x[0]
        else:
            offset_mask = self.conv_offset_mask(x)
        o1, o2, mask = paddle.chunk(offset_mask, 3, axis=1)
        offset = paddle.concat((o1, o2), axis=1)
        mask = F.sigmoid(mask)

        offset_mean = paddle.mean(paddle.abs(offset))
        if offset_mean > max(x.shape[2:]):
            logger.warning('Offset mean is {}, larger than max(h, w).'.format(offset_mean))

        out = deform_conv2d(x,
                            offset=offset,
                            weight=self.weight,
                            bias=self.bias,
                            stride=self.stride,
                            padding=self.padding,
                            dilation=self.dilation,
                            mask=mask
                            )


        return out