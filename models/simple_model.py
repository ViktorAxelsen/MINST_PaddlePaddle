# -*- coding: utf-8 -*-

import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, FC
import numpy as np
import os
from PIL import Image

class SIMPLE_MODEL(fluid.dygraph.Layer):
    def __init__(self, name_scope):
        super(SIMPLE_MODEL, self).__init__(name_scope)
        self.conv1 = Conv2D('conv2d', num_filters=16, filter_size=5, stride=1, padding=2, act='relu')
        self.conv2 = Conv2D('conv2d', num_filters=32, filter_size=5, stride=1, padding=2, act='relu')
        self.pool2d = Pool2D('pool2d', pool_size=2, pool_stride=2, pool_type='max')
        # self.bn1 = fluid.layers.batch_norm(inputs=16)
        # self.bn2 = fluid.layers.batch_norm(inputs=32)
        self.fc = FC('fc', size=10, act='softmax')
        
    # 定义网络结构的前向计算过程
    def forward(self, inputs, label=None):
        # [b, 1, 28, 28]
        # print("初始shape => ", inputs.shape)
        inputs = self.conv1(inputs)
        # [b, 16, 28, 28]
        # print("shape => ", inputs.shape)
        # inputs = self.bn1(inputs)
        inputs = self.pool2d(inputs)
        # [b, 16, 14, 14]
        # print("shape => ", inputs.shape)
        inputs = self.conv2(inputs)
        # [b, 32, 14, 14]
        # print("shape => ", inputs.shape)
        # inputs = self.bn2(inputs)
        inputs = self.pool2d(inputs)
        # [b, 32, 7, 7]
        # print("shape => ", inputs.shape)
        inputs = fluid.layers.reshape(inputs, shape=[inputs.shape[0], -1])
        # [b, 32 * 7 * 7]
        # print("shape => ", inputs.shape)
        outputs = self.fc(inputs)
        # [b, 10]
        # print("输出shape => ", outputs.shape)
        if label is not None:
            acc = fluid.layers.accuracy(input=outputs, label=label)
            return outputs, acc
        else:
            return outputs
   
        
def simple_model():
    model = SIMPLE_MODEL('simple')
    
    return model
        
        
        
if __name__ == '__main__':
    model = simple_model()
    print(model)
    