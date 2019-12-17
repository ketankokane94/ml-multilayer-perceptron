import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import json, codecs

from scipy.special import softmax

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)



def save_model(fileName, params):
    with open(fileName, 'w') as f:
        json.dump(params, f, cls=NumpyEncoder)


def load_model(fileName):
    obj_text = codecs.open(fileName, 'r', encoding='utf-8').read()
    json_load = json.loads(obj_text)
    params = {}
    params['w1'] = np.asarray(json_load["w1"])
    params['b1'] = np.asarray(json_load["b1"])
    params['w2'] = np.asarray(json_load["w2"])
    params['b2'] = np.asarray(json_load["b2"])
    return params

class MLP:
    def __init__(self, lr = 1e-3, hidden_nodes=8, epochs=1000, batch_size = 4):
        self.lr = lr
        self.hidden_nodes = hidden_nodes
        self.epochs = epochs
        self.batch_size = batch_size

    def set_params(self, params):
        self.params = params

    def relu(self, z):
        return np.maximum(0,z)


    def relu_backward(self, z):
        return 1 * (z > 0)

    def initialize_weight(self, num_features, output_dims, hidden_nodes):
        param = {}
        param['w1'] = np.random.randn(num_features, hidden_nodes)
        param['b1'] = np.random.randn(1, hidden_nodes)
        param['w2'] = np.random.randn(hidden_nodes, output_dims)
        param['b2'] = np.random.randn(1, output_dims)
        return param

    def forward(self, X, params):
        forward_pass = {}
        z1 = np.dot(X, params['w1']) + params['b1']
        a1 = self.relu(z1)
        z2 = np.dot(a1, params['w2']) + params['b2']
        a2 = softmax(z2,axis = 1)

        forward_pass['z1'] = z1
        forward_pass['a1'] = a1
        forward_pass['z2'] = z2
        forward_pass['a2'] = a2
        return forward_pass

    def backprop(self, X, y, model, params):
        yhat = np.copy(model['a2'])
        yhat[range(X.shape[0]), y] -= 1
        dw2 = np.dot(model['a1'].T, yhat)
        db2 = np.sum(yhat, axis = 0, keepdims=True)
        yhat2 = np.dot(yhat, params['w2'].T) * self.relu_backward(model['a1'])
        dw1 = np.dot(X.T, yhat2)
        db1 = np.sum(yhat2, axis = 0)
        update = {}
        update['dw2'] = dw2
        update['db2'] = db2
        update['dw1'] = dw1
        update['db1'] = db1

        return update

    def update_weight(self, params , weight_update):
        lr = self.lr
        params['w1'] = params['w1'] - lr * weight_update['dw1']
        params['b1'] = params['b1'] - lr * weight_update['db1']
        params['w2'] = params['w2'] - lr * weight_update['dw2']
        params['b2'] = params['b2'] - lr * weight_update['db2']
        return params

    def cost(self,yhat, y):
        yh = yhat[range(len(yhat)), y]
        return -(np.sum(np.log(yh)))/y.shape[0]


    def fit(self,X, y):
        params = self.initialize_weight(X.shape[1], len(np.unique(y)), self.hidden_nodes)
        self.c = []
        for epoch in range(self.epochs):
            printCost = True
            X, y = shuffle(X,y)
            forward_pass = self.forward(X, params)
            weight_updates = self.backprop(X, y, forward_pass, params)
            params = self.update_weight(params, weight_updates)
            if epoch % 50 == 0 and printCost:
                printCost  = False
                self.c.append(self.cost(forward_pass['a2'], y))
                print('epoch #', epoch,self.c[-1])
        self.params = params
        return params

    def fit_with_validation(self, X_train, y_train, X_validate, y_validate):
        params = self.initialize_weight(X_train.shape[1], len(np.unique(y_train)), self.hidden_nodes)
        self.loss = []
        self.vloss = []
        batch = self.batch_size
        for _ in range(self.epochs):
            # shuffle the entire data set on every epochs
            X_train, y_train = shuffle(X_train, y_train)
            X_validate, y_validate = shuffle(X_validate, y_validate)
            printError = True
            for idx in range(0, X_train.shape[0], batch):
                X = X_train[idx:idx + 1 * batch].reshape(-1,X_train.shape[1])
                y = np.array(y_train[idx: idx + 1 * batch]).reshape(-1)
                forward_pass = self.forward(X, params)
                weight_updates = self.backprop(X, y, forward_pass, params)
                if printError:
                    printError = False
                    self.loss.append(self.cost(forward_pass['a2'], y))
                    forward_pass = self.forward(X_validate, params)
                    self.vloss.append(self.cost(forward_pass['a2'], y_validate))
                    print('epoch #', _, self.vloss[-1])
                params = self.update_weight(params, weight_updates)
        self.params = params
        return params

    def predict(self,X):
        forward_pass = self.forward(X, self.params)
        return np.argmax(forward_pass['a2'], axis = 1)

    def plot_cost_function(self):
        plt.plot(self.c)
        plt.title('Cost function VS Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Cost')
        plt.show()

if __name__ == '__main__':
    pass
