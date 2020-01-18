import numpy as np
import argparse
import ast
import paddle
import paddle.fluid as fluid
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, FC
from paddle.fluid.dygraph.base import to_variable
import math

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
            bias_attr=False)

        self._batch_norm = BatchNorm(name_scope='bn', num_channels=num_filters, act=act)

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)

        return y


class BottleneckBlock(fluid.dygraph.Layer):
    def __init__(self,
                 name_scope,
                 num_channels,
                 num_filters,
                 stride,
                 shortcut=True):
        super(BottleneckBlock, self).__init__(name_scope)

        self.conv0 = ConvBNLayer(
            name_scope='bn_layer',
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=1,
            act='relu')
        self.conv1 = ConvBNLayer(
            name_scope='bn_layer',
            num_channels=num_filters,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            act='relu')
        self.conv2 = ConvBNLayer(
            name_scope='bn_layer',
            num_channels=num_filters,
            num_filters=num_filters * 4,
            filter_size=1,
            act=None)

        if not shortcut:
            self.short = ConvBNLayer(
                name_scope='bn_layer',
                num_channels=num_channels,
                num_filters=num_filters * 4,
                filter_size=1,
                stride=stride)

        self.shortcut = shortcut

        self._num_channels_out = num_filters * 4

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)
        conv2 = self.conv2(conv1)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)

        y = fluid.layers.elementwise_add(x=short, y=conv2)

        layer_helper = LayerHelper(self.full_name(), act='relu')
        return layer_helper.append_activation(y)


class ResNet(fluid.dygraph.Layer):
    def __init__(self, name_scope, layers=50, class_dim=10):
        super(ResNet, self).__init__(name_scope)

        self.layers = layers
        supported_layers = [50, 101, 152]
        assert layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(supported_layers, layers)

        if layers == 50:
            depth = [3, 4, 6, 3]
        elif layers == 101:
            depth = [3, 4, 23, 3]
        elif layers == 152:
            depth = [3, 8, 36, 3]
        num_channels = [64, 256, 512, 1024]
        num_filters = [64, 128, 256, 512]

        self.conv = ConvBNLayer(
            name_scope='bn_layer',
            num_channels=3,
            num_filters=64,
            filter_size=7,
            stride=2,
            act='relu')
        self.pool2d_max = Pool2D(
            name_scope='pool2d',
            pool_size=3,
            pool_stride=2,
            pool_padding=1,
            pool_type='max')

        self.bottleneck_block_list = []
        for block in range(len(depth)):
            shortcut = False
            for i in range(depth[block]):
                bottleneck_block = self.add_sublayer(
                    'bb_%d_%d' % (block, i),
                    BottleneckBlock(
                        name_scope='bnb_layer',
                        num_channels=num_channels[block]
                        if i == 0 else num_filters[block] * 4,
                        num_filters=num_filters[block],
                        stride=2 if i == 0 and block != 0 else 1,
                        shortcut=shortcut))
                self.bottleneck_block_list.append(bottleneck_block)
                shortcut = True

        self.pool2d_avg = Pool2D(name_scope='pool2d',
            pool_size=7, pool_type='avg', global_pooling=True)

        self.pool2d_avg_output = num_filters[len(num_filters) - 1] * 4 * 1 * 1

        import math
        stdv = 1.0 / math.sqrt(2048 * 1.0)
        
        self.out1 = FC(name_scope='fc1',
                          size=1024,
                          act='relu',
                          param_attr=fluid.param_attr.ParamAttr(
                              initializer=fluid.initializer.Uniform(-stdv, stdv)))
                              
        self.out2 = FC(name_scope='fc2',
                          size=512,
                          act='relu',
                          param_attr=fluid.param_attr.ParamAttr(
                              initializer=fluid.initializer.Uniform(-stdv, stdv)))

        self.out3 = FC(name_scope='fc3',
                          size=class_dim,
                          act='softmax',
                          param_attr=fluid.param_attr.ParamAttr(
                              initializer=fluid.initializer.Uniform(-stdv, stdv)))

    def forward(self, inputs, label=None):
        # print("初始shape => ", inputs.shape) # [b, 1, 28, 28]
        
        # init_h = fluid.layers.fill_constant( [2, inputs.shape[0], 28], 'float32', 0.0 )
        # init_c = fluid.layers.fill_constant( [2, inputs.shape[0], 28], 'float32', 0.0 )
        # out, h, c = fluid.layers.lstm(input=fluid.layers.reshape(inputs, shape=[-1, 28, 28]), hidden_size=28, num_layers=2, init_c=init_c, init_h=init_h, max_len=128) # [28, b, 28]
        # out = fluid.layers.reshape(out, shape=[-1, 1, 28, 28]) # [b, 1, 28, 28]
        # inputs = fluid.layers.concat(input=[inputs, out], axis=1) # [b, 2, 28, 28]
        
        # print("cat后 shape => ", inputs.shape)
        y = self.conv(inputs)
        # print("1卷积后 shape => ", y.shape)
        y = self.pool2d_max(y)
        # print("2池化后 shape => ", y.shape)
        for bottleneck_block in self.bottleneck_block_list:
            y = bottleneck_block(y)
        # print("3残差块后 shape => ", y.shape)
        y = self.pool2d_avg(y)
        # print("4池化后 shape => ", y.shape)
        y = fluid.layers.reshape(y, shape=[-1, self.pool2d_avg_output])
        # print("5铺平后 shape => ", y.shape)
        y = self.out1(y)
        y = self.out2(y)
        y = self.out3(y)
        # print("输出shape => ", y.shape)
        if label is not None:
            acc = fluid.layers.accuracy(input=y, label=label)
            return y, acc
        else:
            return y
        
        
def ResNet34():
    return ResNet('ResNet34')

def ResNet50():
    return ResNet(name_scope='ResNet50')
    
def ResNet101():
    return ResNet('ResNet101')
    
def ResNet152():
    return ResNet('ResNet152')
        
        
if __name__ == '__main__':
    with fluid.dygraph.guard():
        model = ResNet50()
        x = np.random.rand(64, 1, 28, 28).astype('float32')
        x = fluid.dygraph.to_variable(x)
        model(x)