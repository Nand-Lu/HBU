# HBU
基于pytorch的分类代码，正在逐步完善
pytorch for VGG，Resnet,Inception-Resnet,
CBAM,
Grad-Cam,
ConfusionMatrix,
待增加：

1、混淆矩阵  #已增加

2、预训练（迁移训练）#已增加

3、进行批量预测

4、优化损失函数

5、动态学习率 #已增加

一、配置训练环境

1、`pip install -r requirements.txt`

2、GPU\CPU都可，根据自己设备性能设置batch_size

3、windows环境也可运行

4、保证有5G以上的运行内存

二、文件夹

训练: 运行 train.py

参数设置:hyper_params.py

预测: 运行 predict.py

Result:保存生成的pth文件，可以在train.py中设置路径

model:添加自己的网络模型，在train.py中 net = xxx()切换

热力图: 运行 Grad—_cam.py

三、数据集
 
 放在datasets文件夹下
 
    -data
   
     --train
      ---类别1
      ---类别2
      
     --valid
      ---类别1
      ---类别2
      
     --test

