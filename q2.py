import numpy as np
import pandas as pd
from MLP import MLP
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# from MLP import save_model
# from MLP import load_model


def plot(loss, vloss):
    fig, ax = plt.subplots()
    ax.plot(loss, label = 'traing loss')
    ax.plot(vloss, label = 'validation loss')
    fig.set_size_inches(4, 4)
    ax.set(xlabel='Epochs', ylabel='Cost',
           title='Cost vs Epochs')
    ax.grid()
    legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')


if __name__ == '__main__':

    data = pd.read_csv('mnist_dataset.csv',header=None)
    y = data[0].values
    X = data.drop([0], axis=1).values
    X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.20, stratify=y)
    X_train = (X_train/255.0 * 0.99 ) + 0.01
    X_validate = (X_validate/255.0 * 0.99 ) + 0.01

    mlp = MLP(lr= 0.01, hidden_nodes = 34, epochs= 100, batch_size = 128)
    # generate the model params by fitting the spiral data set
    params = mlp.fit_with_validation(X_train,y_train,X_validate,y_validate)  #uncomment this to train the model again
    # save the model params to a file
    # save_model('q1c_param.json', params)
    #read the model params from a file
    # params = load_model('q1c_param.json')
    mlp.set_params(params)
    # code to generate the classification_report for traing and validation set both
    print('classification_report on training set')
    y_pred = mlp.predict(X_train)
    print(confusion_matrix(y_train, y_pred))
    print(accuracy_score(y_train, y_pred))

    print('classification_report on validation set')
    y_pred = mlp.predict(X_validate)
    print(confusion_matrix(y_validate, y_pred))
    print(accuracy_score(y_validate, y_pred))

    plot(mlp.loss, mlp.vloss)
    plt.show()
