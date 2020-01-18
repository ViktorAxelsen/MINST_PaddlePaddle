import numpy as np
import argparse
import ast
import paddle
import paddle.fluid as fluid
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, FC
from paddle.fluid.dygraph.base import to_variable
import math


class LSTM(fluid.dygraph.Layer):
    def __init__(self, name_scope):
        super(LSTM, self).__init__(name_scope)
        
        
    def forward(self, inputs):
        init_h = fluid.layers.fill_constant( [2, inputs.shape[0], 28], 'float32', 0.0 )
        init_c = fluid.layers.fill_constant( [2, inputs.shape[0], 28], 'float32', 0.0 )
        out, h, c = fluid.layers.lstm(input=fluid.layers.reshape(inputs, shape=[-1, 28, 28]), hidden_size=28, num_layers=2, init_c=init_c, init_h=init_h, max_len=128) # [28, b, 28]
        out = fluid.layers.reshape(out, shape=[-1, 1, 28, 28]) # [b, 1, 28, 28]
        inputs = fluid.layers.concat(input=[inputs, out], axis=1) # [b, 2, 28, 28]
        pass
    
    
    
def lstm():
    return LSTM('lstm')
    
    
    
if __name__ == '__main__':
    with fluid.dygraph.guard():
        model = lstm()
        x = np.random.rand(64, 1, 28, 28).astype('float32')
        x = fluid.dygraph.to_variable(x)
        model(x)