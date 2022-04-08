from keras.datasets import mnist
import numpy as np
import pickle
import matplotlib.pyplot as plt
import function
import nn
import PCA



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


with open('./best_model', 'rb') as f:
    model = pickle.load(f)

param_1 = np.abs(PCA.pca(model.linear1.weight))
param_2 = np.abs(PCA.pca(model.linear2.weight))
plt.imshow(param_1.reshape(28, 28, -1))
plt.show()
plt.imshow(param_2.reshape(16, 32, -1))
plt.show()


_, (X_test, y_test) = mnist.load_data()
X_test = X_test.reshape(-1, 28*28)/255
pred = model.forward(X_test)
test = (np.argmax(pred, 1))
for i in range(10):
    loc = np.array(np.where(y_test == i))
    acc_i = np.sum(test[loc] == i)/loc.shape[1]
    print("acc of number {} = {}".format(i, acc_i))

acc = np.sum(test == y_test)/y_test.shape[0]
print("test acc = {}".format(acc))

