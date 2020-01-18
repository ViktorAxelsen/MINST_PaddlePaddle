# MINST_PaddlePaddle

- models下为模型源码

- config.py为训练参数配置

- main.py为训练、验证、测试入口文件

- data_process.py为数据增强与预处理文件


# 简要方案

## 数据增强

- 采用了多种增强方法，包括色彩扰动、随机裁剪、随机平移、弹性变换

- 最终弹性变换效果最好

## 模型融合

- 分别训练了ResNet50和手写的一个简单模型

- 最后测试时，简单的对两个模型输出结果进行相加并经过Softmax后计算acc


# 可能的改进方案

> 由于就搞了3天，时间比较紧，有些方法还没去尝试

- 采用交叉验证，充分利用训练集与验证集，避免过拟合

- 采用不同的网络结构，例如，加深和加宽的网络结构，或者使用LSTM进行预测，最后将模型进行融合


# 命令

- python main.py train 表示开始训练

- python main.py test_voting 表示对测试集进行模型融合测试

- python main.py test 表示对测试机进行普通单模测试


**若需检测acc，则输入python main.py test_voting即可**
**模型融合准确率为0.9961**