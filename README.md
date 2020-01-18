# MINST_PaddlePaddle

models下为模型源码
config.py为训练参数配置
main.py为训练、验证、测试入口文件
data_process.py为数据增强与预处理文件

# 命令

- python main.py train 表示开始训练

- python main.py test_voting 表示对测试集进行模型融合测试

- python main.py test 表示对测试机进行普通单模测试


**若需检测acc，则输入python main.py test_voting即可**
**模型融合准确率为0.9961**