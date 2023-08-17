from tensorflow.keras.layers import Input,Lambda,Dense
from tensorflow.keras.models import Model
from vgg16 import VGG16

def siamese(input_shape):
    # 基础VGG，因为两个网络共享参数，所以需要是同一个实例
    base_network = VGG16()
    
    left_input = Input(shape = input_shape)
    right_input = Input(shape = input_shape)
    
    left_output = base_network.call(left_input)
    right_output = base_network.call(right_input)
    
    # l1距离
    output = Lambda(lambda x:abs(x[0]-x[1]))([left_output,right_output])
    
    # 全连接层1
    output = Dense(512,activation = 'relu')(output)
    
    # 全连接层2
    output = Dense(1,activation = 'sigmoid')(output)
    
    return Model([left_input,right_input],output)