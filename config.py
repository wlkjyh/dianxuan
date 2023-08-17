""" 
    网络相关配置文件
"""

# 预训练权重
weight_model_path = './vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

# 输入图像大小，52*52*3
input_shape = (52,52,3)

# 如果不使用GPU训练，则传入列表，多个GPU则传入多个编号
gpu = []
# gpu = [0,1,2,3]

# 训练ecoph数
epochs = 3000
# 批次大小
batch_size = 128

# 学习率
lr = 1e-3

# 是否开启tensorboard
tensorboard = True

# tensorboard日志目录
tensorboard_log_dir = './logs'

# 验证集比例
valid_rate = 0.05

sample_path = './sample'

auto_best_checkpoint_path = './checkpoint/best.h5'

auto_epoch_checkpoint_path = './checkpoint/epoch_{epoch:03d}.h5'

model_save_path = './model.h5'