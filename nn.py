import numpy as np
import function
import random


class LinearLayer:

    def __init__(self, in_features, out_features, momentum):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = 0.1 * np.random.randn(self.in_features, self.out_features)
        self.bias = 0.1 * np.random.randn(self.out_features).reshape(1, -1)
        self.weight_previous_direction = np.zeros((in_features, out_features))
        self.bias_previous_direction = np.zeros(self.out_features)
        self.l2 = 0
        self.momentum = momentum
        self.inputs = 0
        self.outputs = 0
        self.grad_bias = 0
        self.grad_weight = 0
        self.delta_input = 0
        self.delta_output = 0
        self.batch_size = 0

    def forward(self, inputs):
        self.inputs = inputs
        self.batch_size = inputs.shape[0]
        self.outputs = np.dot(self.inputs, self.weight) + self.bias
        self.l2 = np.sum(self.weight**2)
        return self.outputs

    def backward(self, delta_output):
        self.grad_weight = np.zeros((self.batch_size, self.in_features, self.out_features))
        self.delta_output = delta_output
        self.grad_weight = np.dot(self.inputs.T, self.delta_output)
        self.grad_bias = self.delta_output
        self.delta_input = np.dot(self.delta_output, self.weight.T)
#        self.update()

    def update(self, lr, weight_decay):
        grad_weight_avg = self.grad_weight / self.batch_size
        grad_bias_avg = np.mean(self.grad_bias, 0)
        self.weight_previous_direction = self.momentum * self.weight_previous_direction + \
                                         lr * (grad_weight_avg + weight_decay * self.weight)
        self.bias_previous_direction = self.momentum * self.bias_previous_direction + \
                                         lr * grad_bias_avg
        self.weight = self.weight - self.weight_previous_direction
        self.bias = self.bias - self.bias_previous_direction


class Activate:

    def __init__(self, function_name):
        if function_name == 'sigmoid':
            self.activation = function.sigmoid
            self.der_activation = function.der_sigmoid
        elif function_name == 'relu':
            self.activation = function.relu
            self.der_activation = function.der_relu
        elif function_name == 'tanh':
            self.activation = function.tanh
            self.der_activation = function.der_tanh
        elif function_name == 'Lrelu':
            self.activation = function.Lrelu
            self.der_activation = function.der_Lrelu
        else:
            raise ValueError("输入激活函数不存在")
        self.inputs = 0
        self.outputs = 0
        self.delta_input = 0

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = self.activation(self.inputs)
        return self.outputs

    def backward(self, delta_output):
        self.delta_input = delta_output * self.der_activation(self.inputs)


class Loss:

    def __init__(self):
        self.loss_function = function.CrossEntrophy_Loss
        self.der_loss = function.der_CrossEntrophy_Loss
        self.inputs = 0
        self.loss = 0
        self.label = 0
        self.delta_inputs = 0

    def forward(self, inputs, label, l2_1, l2_2, weight_decay):
        self.inputs = inputs
        self.label = label
        self.loss = self.loss_function(inputs, label) + \
                    weight_decay * (l2_1 + l2_2)

    def grad_loss_val(self):
        self.delta_inputs = self.der_loss(self.inputs, self.label)


class SGD:
    def __init__(self, linear1, linear2, lr, weight_decay):
        self.weight_decay = weight_decay
        self.lr = lr
        self.linear1 = linear1
        self.linear2 = linear2

    def step(self):
        self.linear1.update(self.lr, self.weight_decay)
        self.linear2.update(self.lr, self.weight_decay)
