
import os
import config
import tensorflow as tf
from siamese import siamese
from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import *
import cv2
import random
import numpy as np
import utils


if len(config.gpu) >= 1:
    # 设置显存分配方式
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in config.gpu)
    print('>>> GPU:',os.environ['CUDA_VISIBLE_DEVICES'])
input_shape = config.input_shape

model = siamese(input_shape=(input_shape[0],input_shape[1],input_shape[2]))

if config.weight_model_path != False:
    """ 
        by_name=True：设置为 True 时，表示根据层的名称进行权重的匹配和加载。这意味着只有具有相同名称的层才会被加载权重，其他层将被忽略。这在微调模型时非常有用，可以只加载与预训练权重文件中的层名称匹配的部分权重。

        skip_mismatch=True：设置为 True 时，表示如果层数量不匹配或找不到对应的层，则跳过权重加载的错误。这在模型的结构发生变化时很有用，可以避免由于层数不匹配而导致的加载错误。请注意，在层数量不匹配的情况下，任何未匹配的层都不会加载权重。
    """
    model.load_weights(config.weight_model_path,by_name=True,skip_mismatch=True)
    
# 如果是多个GPU需要设置
if len(config.gpu) > 1:
    model = multi_gpu_model(model,gpus=len(config.gpu))

opt = SGD(lr=config.lr,momentum=0.9)

min_lr = config.lr * 0.01

nbs             = 64
lr_limit_max    = 1e-3
lr_limit_min    = 3e-4
Init_lr_fit     = min(max(config.batch_size / nbs * config.lr, lr_limit_min), lr_limit_max)
Min_lr_fit      = min(max(config.batch_size / nbs * min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

# 准备数据
sample = os.listdir(config.sample_path)
data = []
hasOne = []
for i in sample:
    t = i.split('_')[0]
    if t not in hasOne:
        hasOne.append(t)
        data.append([1, config.sample_path + '/' + t + '_1.jpg', config.sample_path + '/' + t + '_2.jpg'])
        # 需要做数据增强的部分，到时候会把2转换成1
        data.append([2, config.sample_path + '/' + t + '_1.jpg', config.sample_path + '/' + t + '_2.jpg'])
        data.append([2, config.sample_path + '/' + t + '_1.jpg', config.sample_path + '/' + t + '_2.jpg'])
        data.append([2, config.sample_path + '/' + t + '_1.jpg', config.sample_path + '/' + t + '_2.jpg'])
        
        # 随机负样本
        for j in range(3):
            f = random.choice(sample)
            while f.split('_')[0] == t:
                f = random.choice(sample)
                
            data.append([0, config.sample_path + '/' + t + '_1.jpg', config.sample_path + '/' + f])

# 打乱数据
random.shuffle(data)

train_x_left = np.zeros((len(data),input_shape[0],input_shape[1],input_shape[2]))
train_x_right = np.zeros((len(data),input_shape[0],input_shape[1],input_shape[2]))

train_y = np.zeros((len(data),1))

total = len(data)
now = 0
for i in data:
    now += 1 
    left = cv2.imread(i[1])
    left = cv2.resize(left,(input_shape[0],input_shape[1]))
    
    right = cv2.imread(i[2])
    right = cv2.resize(right,(input_shape[0],input_shape[1]))
    
    label = i[0]
    
    # 数据增强操作
    if label == 2:
        label = 1
        # 随机翻转
        if random.random() > 0.5:
            left = cv2.flip(left,1)
            right = cv2.flip(right,1)
        
        # 随机噪声
        for h in range(5):
            # 加上随机线条
            if random.random() > 0.5:
                cv2.line(left,(random.randint(0,input_shape[0]),random.randint(0,input_shape[1])),(random.randint(0,input_shape[0]),random.randint(0,input_shape[1])),(0,0,0),random.randint(1,3))
                cv2.line(right,(random.randint(0,input_shape[0]),random.randint(0,input_shape[1])),(random.randint(0,input_shape[0]),random.randint(0,input_shape[1])),(0,0,0),random.randint(1,3))
                
            # 加上随机点
            if random.random() > 0.5:
                cv2.circle(left,(random.randint(0,input_shape[0]),random.randint(0,input_shape[1])),random.randint(1,3),(0,0,0),-1)
                cv2.circle(right,(random.randint(0,input_shape[0]),random.randint(0,input_shape[1])),random.randint(1,3),(0,0,0),-1)
            
        # 色域变换
        if random.random() > 0.5:
            left = cv2.cvtColor(left,cv2.COLOR_BGR2HSV)
            right = cv2.cvtColor(right,cv2.COLOR_BGR2HSV)
                
            left[:,:,0] = left[:,:,0] + random.randint(-10,10)
            right[:,:,0] = right[:,:,0] + random.randint(-10,10)
                
            left = cv2.cvtColor(left,cv2.COLOR_HSV2BGR)
            right = cv2.cvtColor(right,cv2.COLOR_HSV2BGR)
            
        if not os.path.exists('./1.jpg'):
            cv2.imwrite('./1.jpg',left)
            cv2.imwrite('./2.jpg',right)
    
    
    # 归一化
    left = left / 255.0
    right = right / 255.0
     
    train_x_left[now-1] = left
    train_x_right[now-1] = right
    train_y[now-1] = label
    
    
    
    print('>>> 正在处理第',now,'/',total,'张图片',end='\r')
    

lr_scheduler_func = utils.get_lr_scheduler('cos', Init_lr_fit, Min_lr_fit, config.epochs)

if config.tensorboard:
    tb = TensorBoard(log_dir=config.tensorboard_log_dir,write_graph=True,write_images=True)
    
best_checkpoint = ModelCheckpoint(config.auto_best_checkpoint_path,save_weights_only=False,save_best_only=True,verbose=1,period=1)
epoch_checkpoint = ModelCheckpoint(config.auto_epoch_checkpoint_path,save_weights_only=False,save_best_only=False,verbose=1,period=10)
lr_scheduler = LearningRateScheduler(lr_scheduler_func, verbose=1)

callback = [best_checkpoint,epoch_checkpoint,lr_scheduler]

if config.tensorboard:
    callback.append(tb)
    

model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['binary_accuracy'])


model.fit([train_x_left,train_x_right],train_y,batch_size=config.batch_size,epochs=config.epochs,validation_split=config.valid_rate,callbacks=callback)
model.save_weights(config.model_save_path)