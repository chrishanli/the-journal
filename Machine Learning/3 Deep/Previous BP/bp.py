# coding:utf-8
import random
import math


#
#   参数解释：
#   "pd_" ：偏导的前缀
#   "d_" ：导数的前缀
#   "wrt_" ：分数线
#   "w_ho" ：隐含层到输出层的权重系数索引
#   "w_ih" ：输入层到隐含层的权重系数的索引

class NeuralNetwork:
    # 学习率
    LEARNING_RATE = 0.5

    def __init__(self,
                 # 输入层神经元数量，隐藏层神经元数量，输出层神经元数量
                 num_inputs, num_hidden, num_outputs,
                 # 隐藏层各初始权重（w），隐藏层所有神经元的偏移量（b）
                 hidden_layer_weights=None, hidden_layer_bias=None,
                 # 输出层各初始权重（w），输出层所有神经元的偏移量（b）
                 output_layer_weights=None, output_layer_bias=None):
        self.num_inputs = num_inputs

        # 新建隐藏层、输出层（各一层）并且设置偏移量（b）
        self.hidden_layer = NeuronLayer(num_hidden, hidden_layer_bias)
        self.output_layer = NeuronLayer(num_outputs, output_layer_bias)

        # 根据提供的初始权重，新建各层间的连接
        self.init_weights_from_inputs_to_hidden_layer_neurons(hidden_layer_weights)
        self.init_weights_from_hidden_layer_neurons_to_output_layer_neurons(output_layer_weights)

    # 初始化输入层到隐藏层的连接及其权重
    def init_weights_from_inputs_to_hidden_layer_neurons(self, hidden_layer_weights):
        weight_num = 0
        # 每个隐藏层，都会和所有输入层的神经元相连
        if not hidden_layer_weights:
            for h in range(len(self.hidden_layer.neurons)):
                for _ in range(self.num_inputs):
                    # 如果没有给出初始权重，就自己整一个随机数出来
                    self.hidden_layer.neurons[h].weights.append(random.random())
                    weight_num += 1
        else:
            for h in range(len(self.hidden_layer.neurons)):
                for _ in range(self.num_inputs):
                    # 为方便起见，权重都记载在目的神经元（隐藏层的神经元）内
                    self.hidden_layer.neurons[h].weights.append(hidden_layer_weights[weight_num])
                    weight_num += 1

    # 初始化隐藏层到输出层的连接及其权重
    def init_weights_from_hidden_layer_neurons_to_output_layer_neurons(self, output_layer_weights):
        weight_num = 0
        # 每个输出层，都会和所有隐藏层的神经元相连
        if not output_layer_weights:
            for o in range(len(self.output_layer.neurons)):
                for h in range(len(self.hidden_layer.neurons)):
                    # 如果没有给出初始权重，就自己整一个随机数出来
                    self.output_layer.neurons[o].weights.append(random.random())
                    weight_num += 1
        else:
            for o in range(len(self.output_layer.neurons)):
                for h in range(len(self.hidden_layer.neurons)):
                    self.output_layer.neurons[o].weights.append(output_layer_weights[weight_num])
                    weight_num += 1

    def inspect(self):
        print('------')
        print('* Inputs: {}'.format(self.num_inputs))
        print('------')
        print('Hidden Layer')
        self.hidden_layer.inspect()
        print('------')
        print('* Output Layer')
        self.output_layer.inspect()
        print('------')

    # 采用 input 的数据集作为输入做一次前向传播
    def feed_forward(self, inputs):
        hidden_layer_outputs = self.hidden_layer.feed_forward(inputs)
        return self.output_layer.feed_forward(hidden_layer_outputs)

    # 训练神经元
    def train(self, training_inputs, training_outputs):
        # 1. 进行一次前向传播
        self.feed_forward(training_inputs)

        # 2. 进行反向传播

        # 2. 1. 计算 总误差 对 各输出层神经元的输出 的偏导（用于更新最后一个隐藏层后边的权重）
        pd_errors_wrt_output_neuron_total_net_input = [0] * len(self.output_layer.neurons)
        for o in range(len(self.output_layer.neurons)):
            # ∂E(total)
            # ---------
            # ∂net(第 o 个输出层神经元)
            # 公式(5.6)的前两项，即(5.7)和(5.9)的积
            pd_errors_wrt_output_neuron_total_net_input[o] = \
                self.output_layer.neurons[o].calc_pd_error_wrt_total_net_input(training_outputs[o])

        # 2. 2. 计算 总误差 对 各隐藏层神经元的输入 的偏导（用于更新最后一个隐藏层前边的权重）
        pd_errors_wrt_hidden_neuron_total_net_input = [0] * len(self.hidden_layer.neurons)
        for h in range(len(self.hidden_layer.neurons)):
            # ∂E(total)
            # ---------
            # ∂out(第 h 个隐藏层神经元)
            d_error_wrt_hidden_neuron_output = 0
            # 2. 2. 1. 计算 总误差 对 该隐藏层神经元的【输出】 的偏导
            # 总误差来自每个输出层神经元，均受各隐藏层神经元输出的影响，都要考虑进来
            # ---  ∂E(第 o 个输出层神经元)
            # \    ----------------------
            # /__  ∂out(第 h 个隐藏层神经元)
            #  o
            for o in range(len(self.output_layer.neurons)):
                # 留意：∂E(第 o 个输出层神经元) / ∂out(第 h 个隐藏层神经元)
                # = [∂E(第 o 个输出层神经元) / ∂net(第 o 个输出层神经元)] * [∂net(第 o 个输出层神经元) / ∂out(第 h 个隐藏层神经元)]
                # = [∂E(第 o 个输出层神经元) / ∂net(第 o 个输出层神经元)] * w(由h神经元至o神经元)
                d_error_wrt_hidden_neuron_output += pd_errors_wrt_output_neuron_total_net_input[o] * \
                                                    self.output_layer.neurons[o].weights[h]

            # 2. 2. 2. 计算 总误差 对 该隐藏层神经元的【输入】 的偏导
            pd_errors_wrt_hidden_neuron_total_net_input[h] = \
                d_error_wrt_hidden_neuron_output * \
                self.hidden_layer.neurons[h]\
                    .calc_pd_output_wrt_net_input()

        # 2. 3. 更新【隐藏层 --> 输出层】的权重系数
        for o in range(len(self.output_layer.neurons)):
            for w_ho in range(len(self.output_layer.neurons[o].weights)):
                # ∂E(total) /∂wᵢ = ∂E(total)/∂net(o) * ∂net(o)/∂wᵢ
                pd_error_wrt_weight = pd_errors_wrt_output_neuron_total_net_input[o] * \
                                      self.output_layer.neurons[o].calc_pd_total_net_input_wrt_weight(w_ho)

                # 调整参数的方法：wᵢ = wᵢ - α(学习率) * ∂E(total) / ∂wᵢ
                self.output_layer.neurons[o].weights[w_ho] -= self.LEARNING_RATE * pd_error_wrt_weight

        # 2. 4. 更新【输入层 --> 隐藏层】的权重系数
        for h in range(len(self.hidden_layer.neurons)):
            for w_ih in range(len(self.hidden_layer.neurons[h].weights)):
                # ∂E(total) /∂wᵢ = ∂E(total)/∂net(h) * ∂net(h)/∂wᵢ
                pd_error_wrt_weight = pd_errors_wrt_hidden_neuron_total_net_input[h] * self.hidden_layer.neurons[
                    h].calc_pd_total_net_input_wrt_weight(w_ih)

                # 调整参数的方法：wᵢ = wᵢ - α(学习率) * ∂E(total) / ∂wᵢ
                self.hidden_layer.neurons[h].weights[w_ih] -= self.LEARNING_RATE * pd_error_wrt_weight

    # 计算当前总误差
    def calc_total_error(self, training_sets):
        total_error = 0
        for training_pair in training_sets:
            training_inputs, training_outputs = training_pair
            # 做一次前馈
            self.feed_forward(training_inputs)
            # 比照计算总误差（每个输出神经元的误差之和）
            for o in range(len(training_outputs)):
                total_error += self.output_layer.neurons[o].calc_error(training_outputs[o])
        return total_error


class NeuronLayer:
    def __init__(self, num_neurons, bias):

        # 同一层的神经元共享一个截距项 b
        self.bias = bias if bias else random.random()

        # 根据要求新建神经元
        self.neurons = []
        for _ in range(num_neurons):
            self.neurons.append(Neuron(self.bias))

    def inspect(self):
        print('Neurons:', len(self.neurons))
        for n in range(len(self.neurons)):
            print(' Neuron', n)
            for w in range(len(self.neurons[n].weights)):
                print('  Weight:', self.neurons[n].weights[w])
            print('  Bias:', self.bias)

    def feed_forward(self, inputs):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.calc_output(inputs))
        return outputs

    def get_outputs(self):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.output)
        return outputs


class Neuron:
    def __init__(self, bias):
        self.bias = bias
        self.weights = []
        self.inputs = []
        self.output = 0.0

    def calc_output(self, inputs):
        self.inputs = inputs
        self.output = self.sigmoid(self.calc_total_net_input())
        return self.output

    def calc_total_net_input(self):
        total = 0
        for i in range(len(self.inputs)):
            total += self.inputs[i] * self.weights[i]
        return total + self.bias

    def sigmoid(self, total_net_input):
        return 1 / (1 + math.exp(-total_net_input))

    def calc_pd_error_wrt_total_net_input(self, target_output):
        return self.calc_pd_error_wrt_output(target_output) * self.calc_pd_output_wrt_net_input()

    def calc_error(self, target_output):
        # 注：每一个神经元的误差是由平方差公式计算的
        return 0.5 * (target_output - self.output) ** 2

    def calc_pd_error_wrt_output(self, target_output):
        return -(target_output - self.output)

    def calc_pd_output_wrt_net_input(self):
        return self.output * (1 - self.output)

    def calc_pd_total_net_input_wrt_weight(self, index):
        return self.inputs[index]


if __name__ == '__main__':

    # 文中的例子:
    nn = NeuralNetwork(num_inputs=2,
                       num_hidden=2,
                       num_outputs=2,
                       hidden_layer_weights=[0.15, 0.2, 0.25, 0.3],
                       hidden_layer_bias=0.35,
                       output_layer_weights=[0.4, 0.45, 0.5, 0.55],
                       output_layer_bias=0.6)
    for i in range(10000):
        nn.train([0.05, 0.1], [0.01, 0.09])
        print(i, round(nn.calc_total_error([[[0.05, 0.1], [0.01, 0.09]]]), 9))

    # 另外一个例子，可以把上面的例子注释掉再运行一下:
    # training_sets = [
    #     [[0, 0], [0]],
    #     [[0, 1], [1]],
    #     [[1, 0], [1]],
    #     [[1, 1], [0]]
    # ]
    #
    # nn = NeuralNetwork(len(training_sets[0][0]), 5, len(training_sets[0][1]))
    # for i in range(10000):
    #     training_inputs, training_outputs = random.choice(training_sets)
    #     nn.train(training_inputs, training_outputs)
    #     print(i, nn.calc_total_error(training_sets))
