# -*- coding: utf-8 -*-

import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, FC
import numpy as np
import os
from PIL import Image

class MODEL(fluid.dygraph.Layer):
    def __init__(self, name_scope):
        super(MODEL, self).__init__(name_scope)
        self.conv1 = Conv2D('conv2d1_', num_filters=40, filter_size=5, stride=1, padding=2, act='relu')
        self.conv2 = Conv2D('conv2d2_', num_filters=50, filter_size=5, stride=1, padding=2, act='relu')
        self.conv3 = Conv2D('conv2d3_', num_filters=70, filter_size=2, stride=1, padding=1, act='relu')
        self.conv4 = Conv2D('conv2d4_', num_filters=100, filter_size=2, stride=1, padding=1, act='relu')
        self.pool2d1 = Pool2D('pool2d1_', pool_size=2, pool_stride=2, pool_type='max')
        self.pool2d2 = Pool2D('pool2d2_', pool_size=2, pool_stride=2, pool_type='max')
        self.pool2d3 = Pool2D('pool2d3_', pool_size=2, pool_stride=2, pool_type='max')
        self.fc1 = FC('fc1_', size=100, act='relu')
        self.fc2 = FC('fc2_', size=10, act='softmax')
        
    # 定义网络结构的前向计算过程
    def forward(self, inputs, label=None):
        inputs = self.conv1(inputs)
        inputs = self.conv2(inputs)
        inputs = self.pool2d1(inputs)
        inputs = fluid.layers.dropout(inputs, dropout_prob=0.2)
        inputs = self.conv3(inputs)
        inputs = self.pool2d2(inputs)
        inputs = self.conv4(inputs)
        inputs = self.pool2d3(inputs)
        inputs = fluid.layers.reshape(inputs, shape=[inputs.shape[0], -1])
        inputs = self.fc1(inputs)
        inputs = fluid.layers.dropout(inputs, dropout_prob=0.2)
        outputs = self.fc2(inputs)
        
        if label is not None:
            acc = fluid.layers.accuracy(input=outputs, label=label)
            return outputs, acc
        else:
            return outputs
   
        
def ano_model():
    model = MODEL('ano')
    
    return model
        
        
        
if __name__ == '__main__':
    with fluid.dygraph.guard():
        model = ano_model()
        x = np.random.rand(64, 1, 28, 28).astype('float32')
        x = fluid.dygraph.to_variable(x)
        model(x)
    