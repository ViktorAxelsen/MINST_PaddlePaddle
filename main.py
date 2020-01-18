# -*- coding: utf-8 -*-

import paddle
import paddle.fluid as fluid
import numpy as np
import os
from PIL import Image
import models
from models import ResNet50, simple_model, ano_model
from config import config
from data_process import load_data


use_gpu = True
place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()

def train(args):
    print('Now startingt training.......')
    with fluid.dygraph.guard(place):
        model = getattr(models, config.model_name)()
        train_loader = load_data('train', config.batch_size)
        data_loader = fluid.io.DataLoader.from_generator(capacity=5, return_list=True)
        data_loader.set_batch_generator(train_loader, places=place)
        # train_loader = paddle.batch(paddle.dataset.mnist.train(), batch_size=config.batch_size)
        # optimizer = fluid.optimizer.Adam(learning_rate=config.lr)
        optimizer = fluid.optimizer.Adam(learning_rate=fluid.layers.piecewise_decay(boundaries=[15630, 31260], values=[1e-3, 1e-4, 1e-5]), regularization=fluid.regularizer.L2Decay(regularization_coeff=1e-4))
        EPOCH_NUM = config.max_epoch
        best_acc = -1
        for epoch_id in range(EPOCH_NUM):
            model.train()
            for batch_id, data in enumerate(data_loader):
                # image_data = np.array([x[0] for x in data]).astype('float32').reshape(-1, 28, 28)
                # label_data = np.array([x[1] for x in data]).astype('int64').reshape(-1, 1)
                # image_data = np.expand_dims(image_data, axis=1)
                image_data, label_data = data
                # print("data shape => ", image_data.shape)
                # print("label shape => ", label_data.shape)
                image = fluid.dygraph.to_variable(image_data)
                label = fluid.dygraph.to_variable(label_data)
                
                predict, avg_acc = model(image, label)
                loss = fluid.layers.cross_entropy(predict, label)
                # print(loss)
                avg_loss = fluid.layers.mean(loss)
                if batch_id !=0 and batch_id  % 200 == 0:
                    print("epoch: {}, batch: {}, loss is: {}, acc is {}".format(epoch_id, batch_id, avg_loss.numpy(), avg_acc.numpy()))
            
                avg_loss.backward()
                optimizer.minimize(avg_loss)
                model.clear_gradients()
                
            
            fluid.save_dygraph(model.state_dict(), config.model_name + '_current')
            val_acc = val(model)
            if val_acc > best_acc:
                fluid.save_dygraph(model.state_dict(), config.model_name + '_best')
            
            best_acc = max(val_acc, best_acc)
            

        
        
        
        
def val(model):
    with fluid.dygraph.guard(place):
        model.eval()
        val_loader = load_data('valid')
        data_loader = fluid.io.DataLoader.from_generator(capacity=5, return_list=True)
        data_loader.set_batch_generator(val_loader, places=place)
        
        acc_set = []
        avg_loss_set = []
        for batch_id, data in enumerate(data_loader):
            x_data, y_data = data
            img = fluid.dygraph.to_variable(x_data)
            label = fluid.dygraph.to_variable(y_data)
            prediction, acc = model(img, label)
            loss = fluid.layers.cross_entropy(input=prediction, label=label)
            avg_loss = fluid.layers.mean(loss)
            acc_set.append(float(acc.numpy()))
            avg_loss_set.append(float(avg_loss.numpy()))
            
        #计算多个batch的平均损失和准确率
        acc_val_mean = np.array(acc_set).mean()
        avg_loss_val_mean = np.array(avg_loss_set).mean()
        
        print('loss={}, acc={}'.format(avg_loss_val_mean, acc_val_mean))
        
        return acc_val_mean
    



def test_voting(args):
    with fluid.dygraph.guard(place):
        model1 = ResNet50()
        model2 = ano_model()
        model_dict1, _ = fluid.load_dygraph('ResNet50' + '_best')
        model_dict2, _ = fluid.load_dygraph('ano_model' + '_best')
        model1.load_dict(model_dict1)
        model2.load_dict(model_dict2)
        model1.eval()
        model2.eval()
        
        test_loader = load_data('eval')
        data_loader = fluid.io.DataLoader.from_generator(capacity=5, return_list=True)
        data_loader.set_batch_generator(test_loader, places=place)
        
        acc_set = []
        avg_loss_set = []
        for batch_id, data in enumerate(data_loader):
            x_data, y_data = data
            img = fluid.dygraph.to_variable(x_data)
            label = fluid.dygraph.to_variable(y_data)
            out1 = model1(img) # [b, 10]
            out2 = model2(img)
            out = out1 + out2
            out = fluid.layers.softmax(input=out)
            acc = fluid.layers.accuracy(input=out, label=label)
            loss = fluid.layers.cross_entropy(input=out, label=label)
            avg_loss = fluid.layers.mean(loss)
            acc_set.append(float(acc.numpy()))
            avg_loss_set.append(float(avg_loss.numpy()))
            
        #计算多个batch的平均损失和准确率
        acc_val_mean = np.array(acc_set).mean()
        avg_loss_val_mean = np.array(avg_loss_set).mean()
        
        print('loss={}, acc={}'.format(avg_loss_val_mean, acc_val_mean))
        
        
        
def test(args):
    with fluid.dygraph.guard(place):
        model = getattr(models, config.model_name)()
        model_dict, _ = fluid.load_dygraph(config.model_name + '_best')
        model.load_dict(model_dict)
        model.eval()
        test_loader = load_data('eval')
        data_loader = fluid.io.DataLoader.from_generator(capacity=5, return_list=True)
        data_loader.set_batch_generator(test_loader, places=place)
        
        acc_set = []
        avg_loss_set = []
        for batch_id, data in enumerate(data_loader):
            x_data, y_data = data
            img = fluid.dygraph.to_variable(x_data)
            label = fluid.dygraph.to_variable(y_data)
            prediction, acc = model(img, label)
            loss = fluid.layers.cross_entropy(input=prediction, label=label)
            avg_loss = fluid.layers.mean(loss)
            acc_set.append(float(acc.numpy()))
            avg_loss_set.append(float(avg_loss.numpy()))
            
        #计算多个batch的平均损失和准确率
        acc_val_mean = np.array(acc_set).mean()
        avg_loss_val_mean = np.array(avg_loss_set).mean()
        
        print('loss={}, acc={}'.format(avg_loss_val_mean, acc_val_mean))
    
    
    
    
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("command", metavar="<command>", help="train or infer")
    parser.add_argument("--ckpt", type=str, help="the path of model weight file")
    parser.add_argument("--ex", type=str, help="experience name")
    parser.add_argument("--resume", action='store_true', default=False)
    args = parser.parse_args()
    if (args.command == "train"):
        train(args)
    if (args.command == "test"):
        test(args)
    if (args.command == "val"):
        val(args)
    if (args.command == "test_voting"):
        test_voting(args)
