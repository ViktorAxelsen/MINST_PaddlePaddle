# -*- coding: utf-8 -*-

import os


class Config:

    #训练的模型名称
    # model_name = 'ResNet50'
    # model_name = 'simple_model'
    model_name = 'ano_model'
    #训练时的batch大小
    batch_size = 2048
    #label的类别数
    num_classes = 10
    #最大训练多少个epoch
    max_epoch = 30
    #初始的学习率
    lr = 1e-3


config = Config()
