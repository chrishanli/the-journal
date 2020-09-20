import time
import tensorflow_datasets as tf_datasets
import tensorflow as tf
from network import Transformer
from utils import create_padding_mark, create_look_ahead_mark

MAX_LENGTH = 40
BUFFER_SIZE = 20000
BATCH_SIZE = 64
EPOCHS = 1


def filter_max_length(x, y, max_length=MAX_LENGTH):
    return tf.logical_and(tf.size(x) <= max_length,
                          tf.size(y) <= max_length)


# 构建掩码
def create_mask(inputs, targets):
    encode_padding_mask = create_padding_mark(inputs)
    # 这个掩码用于掩输入解码层第二层的编码层输出
    decode_padding_mask = create_padding_mark(inputs)

    # look_ahead 掩码， 掩掉未预测的词
    look_ahead_mask = create_look_ahead_mark(tf.shape(targets)[1])
    # 解码层第一层得到padding掩码
    decode_targets_padding_mask = create_padding_mark(targets)

    # 合并解码层第一层掩码
    combine_mask = tf.maximum(decode_targets_padding_mask, look_ahead_mask)

    return encode_padding_mask, combine_mask, decode_padding_mask


class Dataset:
    def __init__(self):
        # 使用到 Portuguese-English 翻译数据集，内含 50,000 个训练样本，1,100 个验证本及 2,000 个测试本。
        examples, metadata = tf_datasets.load('ted_hrlr_translate/pt_to_en',
                                              with_info=True,
                                              as_supervised=True)
        self.train_examples, self.val_examples = examples['train'], examples['validation']

        # 创建自定义的 subwords tokenizer，用于将那些原本不在词典中的词语劈开成子串，使这些子串都在词典中
        self.tokenizer_en = tf_datasets.features.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for pt, en in self.train_examples), target_vocab_size=2 ** 13)
        self.tokenizer_pt = tf_datasets.features.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, en in self.train_examples), target_vocab_size=2 ** 13)

        self.train_dataset = None
        self.val_dataset = None
        self.pt_batch = None
        self.en_batch = None

    def build(self):
        # 使用.map()运行相关图操作
        train_dataset = self.train_examples.map(self.tf_encode)
        # 过滤过长的数据
        train_dataset = train_dataset.filter(filter_max_length)
        # 为数据创建缓存，使数据可以被加速读入
        train_dataset = train_dataset.cache()
        # 打乱并获取批数据
        train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE, padded_shapes=([40], [40]))
        # 设置预取数据
        self.train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        val_dataset = self.val_examples.map(self.tf_encode)
        # 设置验证集数据
        self.val_dataset = val_dataset.filter(filter_max_length).padded_batch(BATCH_SIZE, padded_shapes=([40], [40]))

        self.pt_batch, self.en_batch = next(iter(train_dataset))
        return self

    def encode(self, lang1, lang2):
        lang1 = [self.tokenizer_pt.vocab_size] + self.tokenizer_pt.encode(lang1.numpy()) + \
                [self.tokenizer_pt.vocab_size + 1]

        lang2 = [self.tokenizer_en.vocab_size] + self.tokenizer_en.encode(lang2.numpy()) + \
                [self.tokenizer_en.vocab_size + 1]
        return lang1, lang2

    def tf_encode(self, pt, en):
        return tf.py_function(self.encode, [pt, en], [tf.int64, tf.int64])


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def get_config(self):
        pass

    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


class Train:
    def __init__(self):
        self.num_layers = 4
        self.d_model = 128
        self.dff = 512
        self.num_heads = 8

        self.dataset = Dataset().build()
        self.input_vocab_size = self.dataset.tokenizer_pt.vocab_size + 2
        self.target_vocab_size = self.dataset.tokenizer_en.vocab_size + 2
        self.max_seq_len = 40
        self.dropout_rate = 0.1
        self.learning_rate = CustomSchedule(self.d_model)
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate,
                                                  beta_1=0.9,
                                                  beta_2=0.98,
                                                  epsilon=1e-9)
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                         reduction='none')
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        self.transformer = Transformer(self.num_layers,
                                       self.d_model,
                                       self.num_heads,
                                       self.dff,
                                       self.input_vocab_size,
                                       self.target_vocab_size,
                                       self.max_seq_len,
                                       self.dropout_rate)
        # checkpoint管理器
        checkpoint = tf.train.Checkpoint(transformer=self.transformer,
                                         optimizer=self.optimizer)
        checkpoint_path = './check'
        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_path, max_to_keep=3)
        if self.checkpoint_manager.latest_checkpoint:
            checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print('last checkpoint restore')

    def train(self):
        # 开始训练
        for epoch in range(EPOCHS):
            start = time.time()

            # 重置记录项
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()

            # inputs 葡萄牙语， targets英语
            for batch, (inputs, targets) in enumerate(self.dataset.train_dataset):
                # 训练
                self.train_step(inputs, targets)
                if batch % 20 == 0:
                    print('epoch {}, batch {}, loss:{:.4f}, acc:{:.4f}'.format(
                        epoch + 1, batch, self.train_loss.result(), self.train_accuracy.result()
                    ))

            if (epoch + 1) % 2 == 0:
                checkpoint_save_path = self.checkpoint_manager.save()
                print('epoch {}, save model at {}'.format(
                    epoch + 1, checkpoint_save_path
                ))

            print('epoch {}, loss:{:.4f}, acc:{:.4f}'.format(
                epoch + 1, self.train_loss.result(), self.train_accuracy.result()
            ))

            print('time in 1 epoch:{} secs\n'.format(time.time() - start))

    @tf.function
    def train_step(self, inputs, targets):
        tar_inp = targets[:, :-1]
        tar_real = targets[:, 1:]
        # 构造掩码
        encode_padding_mask, combined_mask, decode_padding_mask = create_mask(inputs, tar_inp)

        with tf.GradientTape() as tape:
            predictions, _ = self.transformer(inputs, tar_inp,
                                              True,
                                              encode_padding_mask,
                                              combined_mask,
                                              decode_padding_mask)
            loss = self.loss_func(tar_real, predictions)
        # 求梯度
        gradients = tape.gradient(loss, self.transformer.trainable_variables)
        # 反向传播
        self.optimizer.apply_gradients(zip(gradients, self.transformer.trainable_variables))

        # 记录loss和准确率
        self.train_loss(loss)
        self.train_accuracy(tar_real, predictions)

    def loss_func(self, y_true, y_pred):
        mask = tf.math.logical_not(tf.math.equal(y_true, 0))  # 为0掩码标1
        loss_ = self.loss_object(y_true, y_pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)


if __name__ == '__main__':
    app = Train()
    app.train()
