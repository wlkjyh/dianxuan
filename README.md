# 点选
基于孪生神经网络实现的点选识别

## 温馨提示
该项目仅供学习研究改进点选验证码的安全性，请勿用于商用或其他带有攻击性质的业务场景中！！！

### 如何使用？
准备vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5预训练权重，网上可以下载到

准备一个yolo分割模型，可以直接训练，具体用途是让他从验证码中分割出问题和点选的字符。

安装环境，我用到的是python3.10
```
pip3 install -r requirement.txt
```

准备数据集，放入data中，格式为 id_序号.jpg|png，id可以采用uuid，序号第一张图是1，第二张图是2，只能两张图

开始训练
```
python train.py
```

预测
```
python predict.py
```
