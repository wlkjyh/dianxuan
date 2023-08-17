import cv2
import numpy as np
from keras.models import load_model
from keras.layers import Lambda
from keras import backend as K
import os
import random
""" 
    孪生网络 对比模型
"""
resize_height, resize_width, channel = 52, 52, 3
weight = "./best.h5"

output = Lambda(lambda x: K.abs(x[0] - x[1]))
model = load_model(weight, custom_objects={'output': output})


image_path = os.listdir("./data")
# 随机选取一张图片
# inp = input('请输入图片名称：')
image_path = "./data/" + random.choice(image_path)
# image_path = './sample/' + inp
weight = "./yolov3-tiny_17000.weights"
cfg = "./yolov3-tiny.cfg"
img = cv2.imread(image_path)
# 加载模型
net = cv2.dnn.readNet(weight, cfg)
height, width, channels = img.shape
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)  # 预处理

net.setInput(blob)
outs = net.forward(net.getUnconnectedOutLayersNames())
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > 0.1:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.1, 0.1)

thickness = 2
color = (0, 255, 0)
font = cv2.FONT_HERSHEY_PLAIN

new_boxes = []

for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]

        new_boxes.append([x, y, w, h])

up_img = sorted(new_boxes, key=lambda x_: x_[1])[0:len(new_boxes)//2]  # 按照y排列 取出上面的
up_img = sorted(up_img, key=lambda x_: x_[0])  # 按照x排序

location_up = {}
for i, j in enumerate(up_img):
    location_up[i+1] = [img[j[1]:j[1]+j[3], j[0]:j[0]+j[2]].astype('float64') / 255.0, j]

down_img = sorted(new_boxes, key=lambda x_: x_[1])[len(new_boxes)//2:]  # 取出下面的
down_img = sorted(down_img, key=lambda x_: x_[0])  # 按照x排序

location_down = {}
for i, j in enumerate(down_img):
    # location[i+1] = j
    location_down[i+1] = [img[j[1]:j[1]+j[3], j[0]:j[0]+j[2]].astype('float64') / 255.0, j]

new_list = []
for down_i, down_img_ in location_down.items():
    #  先是读取下面的图
    temp = []
    for up_i, up_img_ in location_up.items():
        down = np.expand_dims(cv2.resize(down_img_[0], (52, 52)), axis=0)
        up = np.expand_dims(cv2.resize(up_img_[0], (52, 52)), axis=0)
        predict = model.predict([down, up])
        temp.append(predict[0][0])
    temp_ = temp.index(max(temp))
    new_list.append([down_img_[1], temp_])

for (box, pos) in new_list:
    x, y, w, h = box[0], box[1], box[2], box[3]
    x1, y1, x2, y2 = x, y, x + w, y + h
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255))
    cv2.putText(img, str(pos+1), (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1, color, thickness)
cv2.imshow('1', img)
cv2.waitKey(0)
