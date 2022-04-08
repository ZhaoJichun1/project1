import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-1. * x))


def der_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def relu(x):
    return np.where(x > 0, x, 0.)


def der_relu(x):
    return np.where(x < 0, 0., 1.)


def tanh(x):
    return (1 - np.exp(-2 * x)) / (1 + np.exp(-2 * x))


def der_tanh(x):
    return 1 - tanh(x) * tanh(x)


def Lrelu(x):
    return np.where(x < 0, x * 0.1, x)


def der_Lrelu(x):
    return np.where(x < 0, 0.1, 1)


def softmax(inputs):
    e = np.exp(inputs)
    b = np.sum(e, 1)
    batch_size = e.shape[0]
    nod_num = e.shape[1]
    c = np.zeros((batch_size, nod_num))
    for i in range(batch_size):
        c[i, :] = e[i] / b[i]
    return c


def CrossEntrophy_Loss(inputs, label):
    loss = -np.sum(np.log(inputs + 1e-7) * label, 1)
    return loss


def der_CrossEntrophy_Loss(inputs, label):
    delta = inputs - label
    return delta


def onehot(y):
    n = y.shape[0]
    label = np.zeros((n, 10))
    for i in range(n):
        label[i, y[i]] = 1
    return label

