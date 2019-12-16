import numpy as np
import pandas as pd
from MLP import MLP
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from MLP import save_model




if __name__ == '__main__':
    data = pd.read_csv('mnist_train.csv',header=None)
    y = data[0].values
    X = data.drop([0], axis=1).values
    
    X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    
    
    mlp = MLP(lr= 1e-3, hidden_nodes = 25, epochs= 200)
    # generate the model params by fitting the spiral data set
    params = mlp.fit(X,y)  #uncomment this to train the model again
    # save the model params to a file
    # save_model('q1b_param.json', params)
    #read the model params from a file
    # params = load_model('q1b_param.json')
    mlp.set_params(params)
    # code to generate the classification_report and contour plot
    y_pred = mlp.predict(X)

    mlp.plot_cost_function()
    print(classification_report(y, y_pred))
    # draw decision boundary
    plot_decision_boundary(mlp, X, y)
