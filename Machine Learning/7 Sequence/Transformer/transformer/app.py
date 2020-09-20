import tensorflow as tf
from network import Transformer
from train import Dataset, create_mask

MAX_LENGTH = 40


class Translator:
    def __init__(self):
        print('【正在读入数据集】')
        self.dataset = Dataset()
        print('【正在读入网络】')
        latest = tf.train.latest_checkpoint('./check')
        self.transformer = Transformer()
        self.transformer.load_weights(latest)

    def evaluate(self, inp_sentence):
        start_token = [self.transformer.tokenizer_pt.vocab_size]
        end_token = [self.dataset.tokenizer_pt.vocab_size + 1]

        # inp sentence is portuguese, hence adding the start and end token
        inp_sentence = start_token + self.dataset.tokenizer_pt.encode(inp_sentence) + end_token
        encoder_input = tf.expand_dims(inp_sentence, 0)

        # as the target is english, the first word to the transformer should be the
        # english start token.
        decoder_input = [self.dataset.tokenizer_en.vocab_size]
        output = tf.expand_dims(decoder_input, 0)

        for i in range(MAX_LENGTH):
            enc_padding_mask, combined_mask, dec_padding_mask = create_mask(
                encoder_input, output)

            # predictions.shape == (batch_size, seq_len, vocab_size)
            predictions, attention_weights = self.transformer(encoder_input,
                                                              output,
                                                              False,
                                                              enc_padding_mask,
                                                              combined_mask,
                                                              dec_padding_mask)

            # select the last word from the seq_len dimension
            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

            # return the result if the predicted_id is equal to the end token
            if predicted_id == self.dataset.tokenizer_en.vocab_size + 1:
                return tf.squeeze(output, axis=0), attention_weights

            # concatenate the predicted_id to the output which is given to the decoder
            # as its input.
            output = tf.concat([output, predicted_id], axis=-1)

        return tf.squeeze(output, axis=0), attention_weights

    def translate(self, sentence, plot=''):
        result, attention_weights = self.evaluate(sentence)

        predicted_sentence = self.dataset.tokenizer_en.decode([i for i in result
                                                               if i < self.dataset.tokenizer_en.vocab_size])

        print('Input: {}'.format(sentence))
        print('Predicted translation: {}'.format(predicted_sentence))

        if plot:
            pass
