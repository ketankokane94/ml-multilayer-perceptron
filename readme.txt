

q1b.py

Contains code for question 2
Load the data(spiral_train.dat) using pandas and convert it into numpy array

Used the generic implementation of MLP (MLP.py) a single hidden layer network
with configuration learning rate, number of hidden node in the hidden layer and
number of epochs

For training the model called the fit(attr, label) function of MLP to train the model for
the given hyperparameters which ones done training returns the learned parameters.
These parameters are saved in the 'q1b_params.json' file.

The above part is commented once the experimentation with the hyperparameters was completed
and the best parameters where stored in the mentioned json file/

For predicting: I load the params from 'q1b_params.json' and call the
predict(attr) of MLP to get the predicted labels, then to get the accuracy report
used classification_report by sklearn, and also plot the decision boundary

to train the model just uncomment part \
params = mlp.fit(X,y)  #uncomment this to train the model again

which would train the model, store the params in the 'q1b_params.json' and
generate the classification_report and plot_decision_boundary
