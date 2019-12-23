import numpy as np
import pandas as pd
from MLP import MLP
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from MLP import save_model, load_model


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
    data = pd.read_csv('data/spiral_train.dat', header = None)
    # first two columns are the features
    X = data[[0,1]].values
    # column two is the label
    y = data[2].values
    # initialse the model
    mlp = MLP(lr= 1e-3, hidden_nodes = 12, epochs= 20000)
    # generate the model params by fitting the spiral data set
    # params = mlp.fit(X,y)  #uncomment this to train the model again
    # save the model params to a file
    # save_model('q1b_param.json', params)
    #read the model params from a file
    params = load_model('q1b_param.json')
    mlp.set_params(params)
    # code to generate the classification_report and contour plot
    y_pred = mlp.predict(X)
    print(classification_report(y, y_pred))
    # draw decision boundary
    plot_decision_boundary(mlp, X, y)
