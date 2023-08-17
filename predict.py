import cv2
import numpy as np
from keras.models import load_model
from keras.layers import Lambda
from keras import backend as K
import os
import random
from PIL import Image
# 要预测的图片
# image_path = "./sample/1691156257961.jpg"
image_path = os.listdir("./data")
# 随机选取一张图片
# inp = input('请输入图片名称：')
image_path = "./data/" + random.choice(image_path)
# image_path = './sample/' + inp
# print(image_path)

""" 
    YOLOv3 分割模型
"""
weight = "./yolov3-tiny_17000.weights"
cfg = "./yolov3-tiny.cfg"

# 加载模型
net = cv2.dnn.readNet(weight, cfg)
""" 
    孪生网络 对比模型
"""
resize_height, resize_width,channel = 52,52,3

# 自定义的损失和精度
output = Lambda(lambda x: K.abs(x[0] - x[1]))
def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

def binary_accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))
weight = "./best.h5"
# 加载模型
# model = load_model(weight, custom_objects={'contrastive_loss': contrastive_loss, 'binary_accuracy': binary_accuracy})
model = load_model(weight, custom_objects={'output': output})

classes = ["text"]


img = cv2.imread(image_path)

cv2.namedWindow('display')


""" 
    YOLO 分割出内容
"""
height, width, channels = img.shape
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
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

total = len(new_boxes)

print('>>> 检测出：', total,'个字符')

top_total = total / 2
# 如果取出来有小数
if top_total % 1 != 0:
    print('>>> YOLO分割有误，顶部字符数量与点选数量不匹配')
    exit()
    
top_total = int(top_total)

# 取出w最大的（需要点选的）
w_max_boxes = sorted(new_boxes, key=lambda x: x[2], reverse=True)[:top_total]

# 取出剩下的（需要对比的）
w_min_box = sorted(new_boxes, key=lambda x: x[2], reverse=True)[top_total:]

# 按照从左到右排序w_min_box
w_min_box = sorted(w_min_box, key=lambda x: x[0])


w_max_image = []
w_min_image = []

# 分割出具体图像
for i in range(top_total):
    x, y, w, h = w_max_boxes[i]
    p = cv2.resize(img[y:y+h, x:x+w], (resize_height, resize_width))
    
    # cv2.imwrite('./1/1_{}.jpg'.format(i), p)
    
    w_max_image.append(p)
    
for i in range(len(w_min_box)):
    x, y, w, h = w_min_box[i]
    p = cv2.resize(img[y:y+h, x:x+w], (resize_height, resize_width))
    
    # cv2.imwrite('./1/2_{}.jpg'.format(i), p)
    
    w_min_image.append(p)
    
# print(w_max_boxes)

w_max_image_np = np.array(w_max_image) / 255
w_min_image_np = np.array(w_min_image) / 255

select_index = []

# 开始挨个对比，取出最相似的
for i in range(len(w_max_image_np)):
    print('>>> 开始对比第', i+1, '个字符')
    cv2.imwrite('./1/1_{}.jpg'.format(i), w_max_image_np[i] * 255)
    
    left_x = w_max_image_np[i]
    num_index = 0
    cache_rate = 0
    for k in range(len(w_min_image_np)):
        left_y = w_min_image_np[k]
        predict = model.predict([left_x.reshape(1, resize_height, resize_width, channel), left_y.reshape(1, resize_height, resize_width, channel)])
        rate = predict[0][0]
        if rate > cache_rate:
            cv2.imwrite('./1/2_{}.jpg'.format(i), w_min_image_np[k] * 255)
            num_index = k
            
            
        
        cache_rate = rate
        

    select_index.append(num_index)
    
print('>>> 对比完成，结果为：', select_index)

location = []

for i in range(len(select_index)):
    x, y, w, h = w_max_boxes[select_index[i]]
    cv2.rectangle(img, (x, y), (x+w, y+h), color, thickness)
    cv2.putText(img, str(i+1), (x, y+h), font, 1, color, thickness)
    # 转换为图片坐标中心点
    location.append([x+w/2, y+h/2])
    
print('>>> 位置坐标：', location)
    
cv2.imshow('display', img)
cv2.waitKey(0)