import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid import core
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, FC
from paddle.fluid.dygraph.base import to_variable
import sys
import math
import argparse
import ast

class ConvBNLayer(fluid.dygraph.Layer):
    def __init__(self,
                 name_scope,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 groups=1,
                 act=None):
        super(ConvBNLayer, self).__init__(name_scope)

        self._conv = Conv2D(
            name_scope='conv2d',
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            act=None,
            bias_attr=False,
            param_attr=fluid.ParamAttr(name="weights"))

        self._batch_norm = BatchNorm(name_scope='bn', num_channels=num_filters, act=act)

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)

        return y


class SqueezeExcitation(fluid.dygraph.Layer):
    def __init__(self, name_scope, num_channels, reduction_ratio):

        super(SqueezeExcitation, self).__init__(name_scope)
        self._num_channels = num_channels
        self._pool = Pool2D(name_scope='pool2d', pool_size=0, pool_type='avg', global_pooling=True)
        stdv = 1.0 / math.sqrt(num_channels * 1.0)
        self._fc = FC(
            'fc',
            num_channels,
            num_channels // reduction_ratio,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Uniform(-stdv, stdv)),
            act='relu')
        stdv = 1.0 / math.sqrt(num_channels / 16.0 * 1.0)
        self._excitation = FC(
            'fc',
            num_channels // reduction_ratio,
            num_channels,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Uniform(-stdv, stdv)),
            act='sigmoid')

    def forward(self, input):
        y = self._pool(input)
        y = fluid.layers.reshape(y, shape=[-1, self._num_channels])
        y = self._fc(y)
        y = self._excitation(y)
        y = fluid.layers.elementwise_mul(x=input, y=y, axis=0)
        return y


class BottleneckBlock(fluid.dygraph.Layer):
    def __init__(self,
                 name_scope,
                 num_channels,
                 num_filters,
                 stride,
                 cardinality,
                 reduction_ratio,
                 shortcut=True):
        super(BottleneckBlock, self).__init__(name_scope)

        self.conv0 = ConvBNLayer(
            name_scope='convbn',
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=1,
            act="relu")
        self.conv1 = ConvBNLayer(
            name_scope='convbn',
            num_channels=num_filters,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            groups=cardinality,
            act="relu")
        self.conv2 = ConvBNLayer(
            name_scope='convbn',
            num_channels=num_filters,
            num_filters=num_filters * 2,
            filter_size=1,
            act=None)

        self.scale = SqueezeExcitation(
            name_scope='se',
            num_channels=num_filters * 2,
            reduction_ratio=reduction_ratio)

        if not shortcut:
            self.short = ConvBNLayer(
                name_scope='convbn',
                num_channels=num_channels,
                num_filters=num_filters * 2,
                filter_size=1,
                stride=stride)

        self.shortcut = shortcut

        self._num_channels_out = num_filters * 2

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)
        conv2 = self.conv2(conv1)
        scale = self.scale(conv2)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)

        y = fluid.layers.elementwise_add(x=short, y=scale, act='relu')
        return y


class SeResNeXt(fluid.dygraph.Layer):
    def __init__(self, name_scope, layers=50, class_dim=10):
        super(SeResNeXt, self).__init__(name_scope)

        self.layers = layers
        supported_layers = [50, 101, 152]
        assert layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(supported_layers, layers)

        if layers == 50:
            cardinality = 32
            reduction_ratio = 16
            depth = [3, 4, 6, 3]
            num_filters = [128, 256, 512, 1024]
            self.conv0 = ConvBNLayer(
                name_scope='convbn',
                num_channels=3,
                num_filters=64,
                filter_size=7,
                stride=2,
                act='relu')
            self.pool = Pool2D(
                name_scope='pool2d',
                pool_size=3,
                pool_stride=2,
                pool_padding=1,
                pool_type='max')
        elif layers == 101:
            cardinality = 32
            reduction_ratio = 16
            depth = [3, 4, 23, 3]
            num_filters = [128, 256, 512, 1024]
            self.conv0 = ConvBNLayer(
                name_scope='convbn',
                num_channels=3,
                num_filters=64,
                filter_size=7,
                stride=2,
                act='relu')
            self.pool = Pool2D(
                name_scope='pool2d',
                pool_size=3,
                pool_stride=2,
                pool_padding=1,
                pool_type='max')
        elif layers == 152:
            cardinality = 64
            reduction_ratio = 16
            depth = [3, 8, 36, 3]
            num_filters = [128, 256, 512, 1024]
            self.conv0 = ConvBNLayer(
                name_scope='convbn',
                num_channels=3,
                num_filters=64,
                filter_size=3,
                stride=2,
                act='relu')
            self.conv1 = ConvBNLayer(
                name_scope='convbn',
                num_channels=64,
                num_filters=64,
                filter_size=3,
                stride=1,
                act='relu')
            self.conv2 = ConvBNLayer(
                name_scope='convbn',
                num_channels=64,
                num_filters=128,
                filter_size=3,
                stride=1,
                act='relu')
            self.pool = Pool2D(
                name_scope='pool2d',
                pool_size=3,
                pool_stride=2,
                pool_padding=1,
                pool_type='max')

        self.bottleneck_block_list = []
        num_channels = 64
        if layers == 152:
            num_channels = 128
        for block in range(len(depth)):
            shortcut = False
            for i in range(depth[block]):
                bottleneck_block = self.add_sublayer(
                    'bb_%d_%d' % (block, i),
                    BottleneckBlock(
                        name_scope='bnb',
                        num_channels=num_channels,
                        num_filters=num_filters[block],
                        stride=2 if i == 0 and block != 0 else 1,
                        cardinality=cardinality,
                        reduction_ratio=reduction_ratio,
                        shortcut=shortcut))
                num_channels = bottleneck_block._num_channels_out
                self.bottleneck_block_list.append(bottleneck_block)
                shortcut = True

        self.pool2d_avg = Pool2D(
            name_scope='pool2d',
            pool_size=7, pool_type='avg', global_pooling=True)
        stdv = 1.0 / math.sqrt(2048 * 1.0)

        self.pool2d_avg_output = num_filters[len(num_filters) - 1] * 2 * 1 * 1

        self.out = FC('fc', self.pool2d_avg_output,
                      class_dim,
                      param_attr=fluid.param_attr.ParamAttr(
                          initializer=fluid.initializer.Uniform(-stdv, stdv)))

    def forward(self, inputs):
        print("1初始shape => ", inputs.shape) # [b, 1, 28, 28]
        if self.layers == 50 or self.layers == 101:
            y = self.conv0(inputs)
            print("2卷积后shape => ", y.shape)
            y = self.pool(y)
            print("3池化后shape => ", y.shape)
        elif self.layers == 152:
            y = self.conv0(inputs)
            y = self.conv1(y)
            y = self.conv2(y)
            y = self.pool(y)

        for bottleneck_block in self.bottleneck_block_list:
            y = bottleneck_block(y)
        print("4残差SE后shape => ", y.shape)
        y = self.pool2d_avg(y)
        print("5池化后shape => ", y.shape)
        y = fluid.layers.dropout(y, dropout_prob=0.5, seed=100)
        print("6dropout后shape => ", y.shape)
        y = fluid.layers.reshape(y, shape=[-1, self.pool2d_avg_output])
        print("7铺平后shape => ", y.shape)
        y = self.out(y)
        print("8输出shape => ", y.shape)
        return y
        
        
        
def SeResNeXt50():
    return SeResNeXt('SeResNeXt50')
        
        
        
        
if __name__ == '__main__':
    with fluid.dygraph.guard():
        model = SeResNeXt50()
        x = np.random.rand(64, 1, 28, 28).astype('float32')
        x = fluid.dygraph.to_variable(x)
        model(x)