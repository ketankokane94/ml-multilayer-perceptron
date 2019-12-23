import numpy as np
import pandas as pd
from MLP import MLP
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from MLP import save_model, load_model

epsilon = 1e-3

def get_pertubed_weight(idx, flatten_weight, e):
    flatten_weight[idx] = flatten_weight[idx] + e
    return flatten_weight.reshape(2,2)

def get_pertubed_bias(idx, flatten_bias, e):
    flatten_bias[idx] = flatten_bias[idx] + e
    return flatten_bias.reshape(1,2)

def gradient_checking_bias(x,y,params, w, mlp):
    numerical_grad = []
    # because there are 4 weights in the weight matrix
    for idx in range(2):
        # change weight at index idx by adding the epsilon
        params[w] = get_pertubed_bias(idx, params[w].flatten(), epsilon)
        # calculate the output logits
        forward_pass = mlp.forward(x, params)
        # get the Loss using the same Loss function used in softmaxClassifier
        l1 = mlp.cost(forward_pass['a2'], y)
        # change the weight again at index idx this time by subtracting epsilon
        params[w] = get_pertubed_bias(idx, params[w].flatten(), -epsilon)
        params[w] = get_pertubed_bias(idx, params[w].flatten(), -epsilon)
        # calculate the output logits
        forward_pass = mlp.forward(x, params)
        # get the Loss
        l2 = mlp.cost(forward_pass['a2'], y)
        # calculate the slope
        slope =  ((l1 - l2) / (2 * epsilon)) * 4.0
        numerical_grad.append(slope)
        params[w] = get_pertubed_bias(idx, params[w].flatten(), epsilon)
    return numerical_grad

def gradient_checking_weights(x,y,params, w, mlp):
    numerical_grad = []
    # because there are 4 weights in the weight matrix
    for idx in range(4):
        # change weight at index idx by adding the epsilon
        params[w] = get_pertubed_weight(idx, params[w].flatten(), epsilon)
        # calculate the output logits
        forward_pass = mlp.forward(x, params)
        # get the Loss using the same Loss function used in softmaxClassifier
        l1 = mlp.cost(forward_pass['a2'], y)
        # change the weight again at index idx this time by subtracting epsilon
        params[w] = get_pertubed_weight(idx, params[w].flatten(), -epsilon)
        params[w] = get_pertubed_weight(idx, params[w].flatten(), -epsilon)
        # calculate the output logits
        forward_pass = mlp.forward(x, params)
        # get the Loss
        l2 = mlp.cost(forward_pass['a2'], y)
        # calculate the slope
        slope =  ((l1 - l2) / (2 * epsilon)) * 4.0
        numerical_grad.append(slope)
        params[w] = get_pertubed_weight(idx, params[w].flatten(), epsilon)
    return numerical_grad


def gradient_checking(x,y):
    numerical_grad = []
    analytic_grad = []
    mlp = MLP()
    params = mlp.initialize_weight(num_features = 2, output_dims = 2, hidden_nodes = 2)
    mlp.set_params(params)
    numerical_grad.extend(gradient_checking_weights(x,y, params, 'w1', mlp))
    numerical_grad.extend(gradient_checking_bias(x,y, params, 'b1', mlp))
    numerical_grad.extend(gradient_checking_weights(x,y, params, 'w2', mlp))
    numerical_grad.extend(gradient_checking_bias(x,y, params, 'b2', mlp))

    forward_pass = mlp.forward(x, params)
    weight_updates = mlp.backprop(x, y, forward_pass, params)
    analytic_grad.extend(list(weight_updates['dw1'].flatten()))
    analytic_grad.extend(list(weight_updates['db1'].flatten()))
    analytic_grad.extend(list(weight_updates['dw2'].flatten()))
    analytic_grad.extend(list(weight_updates['db2'].flatten()))

    diff = list(zip(numerical_grad,analytic_grad))

    for d in diff:
        print(d[0] , ' & ' , d[1] , ' \\\ \\hline')

def plot_decision_boundary(clf, attr, label):
    h = .01  # step size in the mesh
    # create a mesh to plot in
    x_min, x_max = attr[:, 0].min() - 1, attr[:, 0].max() + 1
    y_min, y_max = attr[:, 1].min() - 1, attr[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.scatter(attr[:,0],attr[:,1], c=label, s = 20 , edgecolor='k')
    plt.contourf(xx, yy, Z, alpha = 0.5)
    plt.show()


if __name__ == '__main__':
    data = pd.read_csv('data/xor.dat', header = None)
    # first two columns are the features
    X = data[[0,1]].values
    # column two is the label
    y = data[2].values
    # initialse the model
    mlp = MLP(lr= 1e-3, hidden_nodes = 8 , epochs= 20000)
    # generate the model params by fitting the spiral data set
    # params = mlp.fit(X,y)  #uncomment this to train the model again
    # save the model params to a file
    # save_model('q1a_param.json', params)
    #read the model params from a file
    params = load_model('q1a_param.json')
    mlp.set_params(params)
    # code to generate the classification_report and contour plot
    y_pred = mlp.predict(X)

    # mlp.plot_cost_function() uncomment this along with fit function to generate the cost function plot
    print(classification_report(y, y_pred))
    # draw decision boundary
    plot_decision_boundary(mlp, X, y)
    # perform gradient checking
    gradient_checking(X, y)
