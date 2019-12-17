import numpy as np
import pandas as pd
from MLP import MLP
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from MLP import save_model
from MLP import load_model


def plot(loss, vloss):
    fig, ax = plt.subplots()
    ax.plot(loss, label = 'traing loss')
    ax.plot(vloss, label = 'validation loss')
    fig.set_size_inches(4, 4)
    ax.set(xlabel='Epochs', ylabel='Cost',
           title='Cost vs Epochs')
    ax.grid()
    legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')

def get_data(fileName):

    data = pd.read_csv(fileName, header = None)
    # first 4 columns are the features
    X = data[[0,1,2,3]].values
    # column 5 is the label
    y= data[4].values
    # initialse the model
    return X, y

if __name__ == '__main__':

    X_train, y_train = get_data('iris_train.dat')
    X_validate, y_validate = get_data('iris_test.dat')

    mlp = MLP(lr= 1e-3, hidden_nodes = 8, epochs= 2000, batch_size = 8)
    # generate the model params by fitting the spiral data set
    # params = mlp.fit_with_validation(X_train,y_train,X_validate,y_validate)  #uncomment this to train the model again
    # save the model params to a file
    # save_model('q1c_param.json', params)
    #read the model params from a file
    params = load_model('q1c_param.json')
    mlp.set_params(params)
    # code to generate the classification_report for traing and validation set both
    print('classification_report on training set')
    y_pred = mlp.predict(X_train)
    print(classification_report(y_train, y_pred))

    print('classification_report on validation set')
    y_pred = mlp.predict(X_validate)
    print(classification_report(y_validate, y_pred))

    # plot(mlp.loss, mlp.vloss)
    # plt.show()
