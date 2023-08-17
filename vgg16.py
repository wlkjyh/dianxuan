from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D
""" 
    VGG16代码来源于https://github.com/bubbliiiing/Siamese-keras
    
    也可以使用tensorflow.keras.applications.VGG16实现
"""

class VGG16:
    def __init__(self):
        # 第一个卷积部分
        # 105, 105, 3 -> 105, 105, 64 -> 52, 52, 64
        self.block1_conv1 = Conv2D(64, (3,3), activation = 'relu', padding = 'same',name = 'block1_conv1')
        self.block1_conv2 = Conv2D(64, (3,3), activation = 'relu', padding = 'same',name = 'block1_conv2')
        self.block1_pool = MaxPooling2D((2,2), strides = (2,2), name = 'block1_pool')
        
        # 第二个卷积部分
        # 52, 52, 64 -> 52, 52, 128 -> 26, 26, 128
        self.block2_conv1 = Conv2D(128, (3,3), activation = 'relu', padding = 'same',name = 'block2_conv1')
        self.block2_conv2 = Conv2D(128, (3,3), activation = 'relu', padding = 'same',name = 'block2_conv2')
        self.block2_pool = MaxPooling2D((2,2), strides = (2,2), name = 'block2_pool')

        # 第三个卷积部分
        # 26, 26, 128-> 26, 26, 256 -> 13, 13, 256
        self.block3_conv1 = Conv2D(256, (3,3), activation = 'relu', padding = 'same',name = 'block3_conv1')
        self.block3_conv2 = Conv2D(256, (3,3), activation = 'relu', padding = 'same',name = 'block3_conv2')
        self.block3_conv3 = Conv2D(256, (3,3), activation = 'relu', padding = 'same',name = 'block3_conv3')
        self.block3_pool = MaxPooling2D((2,2), strides = (2,2), name = 'block3_pool')

        # 第四个卷积部分
        # 13, 13, 256-> 13, 13, 512 -> 6, 6, 512
        self.block4_conv1 = Conv2D(512, (3,3), activation = 'relu', padding = 'same', name = 'block4_conv1')
        self.block4_conv2 = Conv2D(512, (3,3), activation = 'relu', padding = 'same', name = 'block4_conv2')
        self.block4_conv3 = Conv2D(512, (3,3), activation = 'relu', padding = 'same', name = 'block4_conv3')
        self.block4_pool = MaxPooling2D((2,2), strides = (2,2), name = 'block4_pool')

        # 第五个卷积部分
        # 6, 6, 512-> 6, 6, 512 -> 3, 3, 512
        self.block5_conv1 = Conv2D(512, (3,3), activation = 'relu', padding = 'same', name = 'block5_conv1')
        self.block5_conv2 = Conv2D(512, (3,3), activation = 'relu', padding = 'same', name = 'block5_conv2')
        self.block5_conv3 = Conv2D(512, (3,3), activation = 'relu', padding = 'same', name = 'block5_conv3')   
        self.block5_pool = MaxPooling2D((2,2), strides = (2,2), name = 'block5_pool')

        # 3*3*512 = 4500 + 90 + 18 = 4608
        self.flatten = Flatten(name = 'flatten')

    def call(self, inputs):
        x = inputs
        x = self.block1_conv1(x)
        x = self.block1_conv2(x)
        x = self.block1_pool(x)

        x = self.block2_conv1(x)
        x = self.block2_conv2(x)
        x = self.block2_pool(x)

        x = self.block3_conv1(x)
        x = self.block3_conv2(x)
        x = self.block3_conv3(x)
        x = self.block3_pool(x)
        
        x = self.block4_conv1(x)
        x = self.block4_conv2(x)
        x = self.block4_conv3(x)
        x = self.block4_pool(x)

        x = self.block5_conv1(x)
        x = self.block5_conv2(x)
        x = self.block5_conv3(x)
        x = self.block5_pool(x)

        outputs = self.flatten(x)
        return outputs
