"""
简单CNN的训练数据集（采用mnist）和训练流程
Han @ 23/8/2020
"""
import os
import tensorflow as tf
from tensorflow.keras import datasets
from network import CNN


class DataSource(object):
    def __init__(self):
        # 读入mnist数据集，如何不存在将自动下载
        data_path = os.path.abspath(os.path.dirname(__file__)) + '/datasets/mnist.npz'
        (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data(path=data_path)
        # 6万张训练图片，1万张测试图片（全为黑白），输入时进行resize
        train_images = train_images.reshape((60000, 28, 28, 1))
        test_images = test_images.reshape((10000, 28, 28, 1))
        # 像素值从 (0, 255) 映射到 (0, 1) 之间
        train_images, test_images = train_images / 255.0, test_images / 255.0

        self.train_images, self.train_labels = train_images, train_labels
        self.test_images, self.test_labels = test_images, test_labels


class Train:
    def __init__(self):
        # 制作模型，采用adam优化器、交叉熵作为损失函数，并以准确度作为模型训练结果
        self.network = CNN()
        self.network.model.compile(optimizer='adam',
                                   loss='sparse_categorical_crossentropy',
                                   metrics=['accuracy'])
        # 读入数据
        self.data = DataSource()
        print('【网络建立及数据读入已完成】')

    def train(self):
        # 模型要保存到./check_path/cp-xx.chpt文件中
        check_path = './check/cp-{epoch:02d}.chpt'
        # 每隔1 epoch保存一次当前训练的模型的call back，只保存模型权重
        save_model_cb = tf.keras.callbacks.ModelCheckpoint(filepath=check_path,
                                                           save_weights_only=True,
                                                           save_freq='epoch')

        # 开始训练，训练 epochs 个epoch，以train_images作为训练集
        print('【训练中】')
        self.network.model.fit(self.data.train_images,
                               self.data.train_labels,
                               epochs=5,
                               callbacks=[save_model_cb])

        # 测试训练结果
        print('【测试中】')
        test_loss, test_acc = self.network.model.evaluate(self.data.test_images, self.data.test_labels)
        print("【共测试了%d张图片 准确率: %.4f】" % (len(self.data.test_labels), test_acc))


if __name__ == "__main__":
    app = Train()
    app.train()
