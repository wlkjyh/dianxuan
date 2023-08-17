import cv2
import numpy as np
from keras.models import load_model
from keras.layers import Lambda
from keras import backend as K
import os
import random
from PIL import Image

input1 = input('请输入图片名称1：')
input2 = input('请输入图片名称2：')

output = Lambda(lambda x: K.abs(x[0] - x[1]))
weight = "./best.h5"
# 加载模型
# model = load_model(weight, custom_objects={'contrastive_loss': contrastive_loss, 'binary_accuracy': binary_accuracy})
model = load_model(weight, custom_objects={'output': output})


resize = 52
img1 = cv2.imread(input1)
img2 = cv2.imread(input2)

img1 = cv2.resize(img1, (resize, resize)) / 255
img2 = cv2.resize(img2, (resize, resize)) / 255

img1 = np.expand_dims(img1, axis=0)
img2 = np.expand_dims(img2, axis=0)

result = model.predict([img1, img2])

print(result)


