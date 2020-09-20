"""
采用简单CNN的训练数据集（mnist）训练结果（train.py），对自己书写的数字进行识别（预测）
Han @ 23/8/2020
"""
import tensorflow as tf
import numpy as np
from PIL import Image
from train import CNN


class Predict(object):
    def __init__(self):
        print('【正在读入网络】')
        latest = tf.train.latest_checkpoint('./check')
        self.cnn = CNN()
        self.cnn.model.load_weights(latest)

    def predict(self, image_path):
        # 以黑白方式读取图片并缩放
        print('【正读取测试图片：{0}】'.format(image_path))
        img = Image.open(image_path).convert('L')
        if img.size != (28, 28):
            img = img.resize((28, 28))
        # 将读入的图片打扁成向量作为网络输入
        flatten_img = np.reshape(img, (28, 28, 1))

        x = np.array([1 - flatten_img])
        y = self.cnn.model.predict(x)

        # 因为x只传入了一张图片，取y[0]即可
        predict_digit = np.argmax(y[0])
        print('     -> 识别出数字：{0}（猜测：可能性为 {1} %）'.format(predict_digit, y[0][predict_digit] * 100))
        if y[0][predict_digit] < 0.99:
            print('     -> 数字为 0 - 9 的可能性分别为：', y)


if __name__ == "__main__":
    app = Predict()
    for index in range(6):
        app.predict('test_images/{}.png'.format(index))
