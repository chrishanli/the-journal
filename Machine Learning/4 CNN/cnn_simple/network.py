"""
一个比较mini的、tensorflow 支撑的CNN网络
总共有8层
Han @ 23/8/2020
"""

from tensorflow.keras import layers, models


class CNN(object):
    def __init__(self):
        model = models.Sequential()
        # 第1层 卷积，卷积核大小为3*3，32个，28*28为待训练图片的大小
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        # 第2层 池化，方式为MaxPooling，大小为2x2
        model.add(layers.MaxPooling2D((2, 2)))
        # 第3层 卷积，卷积核大小为3*3，64个
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        # 第4层 池化，方式为MaxPooling，大小为2x2
        model.add(layers.MaxPooling2D((2, 2)))
        # 第5层 卷积，卷积核大小为3*3，64个
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))

        # 第6层 扁平化层，将数据变成576个
        model.add(layers.Flatten())
        # 第7层 全连接层（ReLU），将数据变成64维度
        model.add(layers.Dense(64, activation='relu'))
        # 第8层 全连接（分类）层（Softmax），将数据分为10类
        model.add(layers.Dense(10, activation='softmax'))

        self.model = model

    def summary(self):
        self.model.summary()

