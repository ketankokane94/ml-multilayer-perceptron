The generic implementation of MLP which is used by all 4 models
is kept in MLP.py which is imported in all the subsequent py files.

All the cost graph generation code, fit function and save model functions are
commented before submitting the code. As the models are trained and saved in
the appropriate .json files which the model uses to load the weights from to generated
the predictions.

The code required for each question is kept in the required files.

to execute the model simply need to execute the .py file

ex
python3 q1a.py

Execution fails if the required data files and *.json files are not present in the same directory.

To train the model again simply uncomment the call to mlp.fit() function.

For question 2.
Need to unzip the mnist_dataset.zip file and copy the .csv file in the same directory as
q2.py. 
