# -*- coding: utf-8 -*-

import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import FC
import numpy as np
import os
import gzip
import json
import random
import cv2
from PIL import Image, ImageEnhance
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


# 移位
def shift_image(image):
    img = image.reshape(28, 28)
    h, w = img.shape
    key = random.randint(1, 4)
    mat_shift = 0
    if key == 1:
        mat_shift = np.float32([[1, 0, -4], [0, 1, -4]])
    elif key == 2:
        mat_shift = np.float32([[1, 0, 4], [0, 1, -4]])
    elif key == 3:
        mat_shift = np.float32([[1, 0, -4], [0, 1, 4]])
    else:
        mat_shift = np.float32([[1, 0, 4], [0, 1, 4]])
    
    img = cv2.warpAffine(img, mat_shift, (h, w))
    img = img.reshape(1, 28, 28).astype('float32')
    
    return img
    
    
    
# 高斯噪声
def noise_image(image, mean=0, var=0.0005):
    img = image.reshape(28, 28)
    h, w = img.shape
    img = np.array(img / 255.0, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, img.shape)
    out = img + noise
    low_clip = 0
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
        
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out * 255)
    out = out.reshape(1, 28, 28).astype('float32')
    
    return out
    
    
    
# 色彩扰动
def random_color(image, saturation=0, brightness=0, contrast=0, sharpness=0):
    if random.random() < saturation:
        random_factor = np.random.randint(0, 31) / 10.  # 随机因子
        image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
    if random.random() < brightness:
        random_factor = np.random.randint(10, 21) / 10.  # 随机因子
        image = ImageEnhance.Brightness(image).enhance(random_factor)  # 调整图像的亮度
    if random.random() < contrast:
        random_factor = np.random.randint(10, 21) / 10.  # 随机因1子
        image = ImageEnhance.Contrast(image).enhance(random_factor)  # 调整图像对比度
    if random.random() < sharpness:
        random_factor = np.random.randint(0, 31) / 10.  # 随机因子
        ImageEnhance.Sharpness(image).enhance(random_factor)  # 调整图像锐度
    return image
    
    
# 随机仿射弹性变换
def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]

    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
    
    
# 扩充数据集
def expand_dataset(imgs, labels):
    print('Now starting expanding datasets.......')
    for i in range(len(imgs)):
        img = np.reshape(imgs[i], [1, 28, 28]).astype('float32') # [1, 28, 28]
        label = labels[i]
        
        # img_shift = shift_image(img) # 移位后的图像
        # imgs.append(img_shift.reshape(-1).tolist())
        # labels.append(label)
        
        # img_noise = noise_image(img) / 255. # 高斯噪声后的图像
        # imgs.append(img_noise.reshape(-1).tolist())
        # labels.append(label)
        
        # img_rc = np.asarray(random_color(Image.fromarray((img * 255).reshape(28, 28).astype('uint8')), saturation=1, brightness=1, contrast=1, sharpness=1)).reshape(1, 28, 28).astype('float32') / 255. # 色彩扰动后的图像
        # imgs.append(img_rc.reshape(-1).tolist())
        # labels.append(label)
        
        img = (img.reshape(28, 28, 1) * 255).astype('uint8')
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) # [28, 28, 3]
        img_ela = elastic_transform(img, 34, 4, 1)
        img_ela = cv2.cvtColor(img_ela, cv2.COLOR_RGB2GRAY).reshape(1, 28, 28).astype('float32') / 255. # [1, 28, 28]
        imgs.append(img_ela.reshape(-1).tolist())
        labels.append(label)
        
        
        if len(imgs) == 60000:
            print('Datasets have been expanded to => ', len(imgs))
        elif len(imgs) == 80000:
            print('Datasets have been expanded to => ', len(imgs))
    
    print('Datasets have been expanded to => ', len(imgs), ' completely already')
        
        
        


def load_data(mode='train', BATCHSIZE=16):

    # 数据文件
    datafile = './work/mnist.json.gz'
    data = json.load(gzip.open(datafile))
    # 读取到的数据可以直接区分训练集，验证集，测试集
    train_set, val_set, eval_set = data

    # 数据集相关参数，图片高度IMG_ROWS, 图片宽度IMG_COLS
    IMG_ROWS = 28
    IMG_COLS = 28
    # 获得数据
    if mode == 'train':
        imgs = train_set[0]
        labels = train_set[1]
        expand_dataset(imgs, labels)
    elif mode == 'valid':
        imgs = val_set[0]
        labels = val_set[1]
    elif mode == 'eval':
        imgs = eval_set[0]
        labels = eval_set[1]
    else:
        raise Exception("mode can only be one of ['train', 'valid', 'eval']")
        
        
    imgs_length = len(imgs)
    # print(len(imgs), len(labels))

    assert len(imgs) == len(labels), \
          "length of train_imgs({}) should be the same as train_labels({})".format(
                  len(imgs), len(labels))

    index_list = list(range(imgs_length)) # [0, 1, 2, 3, ... 49999]

    # 读入数据时用到的batchsize
    # BATCHSIZE = 100

    # 定义数据生成器
    def data_generator():
        if mode == 'train':
            # 训练模式下，将训练数据打乱
            random.shuffle(index_list)
        imgs_list = []
        labels_list = []
        
        for i in index_list:
            img = np.reshape(imgs[i], [1, IMG_ROWS, IMG_COLS]).astype('float32') # [1, 28, 28]
            label = np.reshape(labels[i], [1]).astype('int64')
            imgs_list.append(img) 
            labels_list.append(label)
            
            if len(imgs_list) == BATCHSIZE:
                # 产生一个batch的数据并返回
                yield np.array(imgs_list), np.array(labels_list)
                # 清空数据读取列表
                imgs_list = []
                labels_list = []

        # 如果剩余数据的数目小于BATCHSIZE，
        # 则剩余数据一起构成一个大小为len(imgs_list)的mini-batch
        if len(imgs_list) > 0:
            yield np.array(imgs_list), np.array(labels_list)
        
        
    return data_generator
    
    
if __name__ == '__main__':
    for batch_id, data in enumerate(load_data('train', 4)()):
        image_data, label_data = data
        # print(image_data[3], label_data)
        break