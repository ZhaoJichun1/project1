from keras.datasets import mnist
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
import function
import nn

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_val = X_train[42000:60000, :, :].reshape(-1, 28*28)/255
y_val = y_train[42000:60000]
X_train = X_train[0:42000, :, :].reshape(-1, 28*28)/255
y_train = y_train[0:42000]
X_test = X_test.reshape(-1, 28*28)/255


y_train_onehot = function.onehot(y_train)
y_val_onehot = function.onehot(y_val)

n_val = X_val.shape[0]
n_train = X_train.shape[0]
n_test = X_test.shape[0]
batch_size = 32
max_iter = 30000

param_grid = {
    'learning_rate': [0.01, 0.1],
    'hidden_size': [16, 128, 512],
    'weight_decay': [0.01, 0.001]
}


class Net:

    def __init__(self, hidden_size):
        self.linear1 = nn.LinearLayer(28*28, hidden_size, 0.9)
        self.activate1 = nn.Activate("relu")
        self.linear2 = nn.LinearLayer(hidden_size, 10, 0.9)
        self.loss = nn.Loss()

    def forward(self, x):
        x = self.linear1.forward(x)
        x = self.activate1.forward(x)
        x = self.linear2.forward(x)
        x = function.softmax(x)
        return x

    def backward(self):
        self.loss.grad_loss_val()
        self.linear2.backward(self.loss.delta_inputs)
        self.activate1.backward(self.linear2.delta_input)
        self.linear1.backward(self.activate1.delta_input)

    def save(self, path):
        obj = pickle.dumps(self)
        with open(path, 'wb') as f:
            f.write(obj)


max_acc = 0
acc = 0

for lr in param_grid["learning_rate"]:
    for hidden_size in param_grid["hidden_size"]:
        for weight_decay in param_grid["weight_decay"]:

            acc_val = []
            loss_val = []
            loss_train = []
            loss_train_sum = 0

            print("\n\nlearning rate = {}, hidden size = {}, weight_decay = {}".format(lr, hidden_size, weight_decay))
            net = Net(hidden_size)
            optimizer = nn.SGD(net.linear1, net.linear2, lr, weight_decay)
            for iter in range(max_iter):
                if iter % 5000 == 0 and iter != 0:
                    if optimizer.lr > 1e-4:
                        optimizer.lr = optimizer.lr/10

                i = random.randint(0, n_train-1)
                if (i + batch_size) > n_train:
                    x_input = np.vstack([X_train[i:n_train, :], X_train[0:(i+batch_size)-n_train, :]])
                    label = np.vstack([y_train_onehot[i:n_train, :], y_train_onehot[0:(i+batch_size)-n_train, :]])
                else:
                    x_input = X_train[i:(i+batch_size), :]
                    label = y_train_onehot[i:(i+batch_size), :]
                output = net.forward(x_input)
                net.loss.forward(output, label, net.linear1.l2, net.linear2.l2, weight_decay)
                net.backward()
                loss_train_sum += np.mean(net.loss.loss)
                optimizer.step()

                if iter % 500 == 0 and iter != 0:
                    loss_train.append(loss_train_sum/500)
                    print(">", end="")
                    y_val_pred = net.forward(X_val)
                    net.loss.forward(y_val_pred, y_val_onehot, net.linear1.l2, net.linear2.l2, weight_decay)
                    validation = (np.argmax(y_val_pred, 1))
                    acc = np.sum(validation == y_val)/n_val
                    acc_val.append(acc)
                    loss = np.mean(net.loss.loss)
                    loss_val.append(loss)
                    if iter % 5000 == 0:
                       print('\nTraining iteration = {}, training loss = {}, validation acc = {}, validation loss = {}'
                             .format(iter, loss_train[-1], acc, loss))
                    loss_train_sum = 0

            if acc > max_acc:
                max_acc = acc
                net.save('./best_model')
                print("\nmodel has been saved")



            y = range(500, 30000, 500)
            plt.figure(1)
            plt.plot(y, loss_train, y, loss_val)
            plt.figure(2)
            plt.plot(y, acc_val)
            plt.show()
