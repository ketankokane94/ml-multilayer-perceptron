# Multi-layer perceptron using Numpy

Implemented a multi-layer (one hidden layer) perceptron capable of non-linear function learning.

## Getting Started

Required data-sets for given examples can be found in the data/ directory. 
Learned params for the toy examples are in params/ directory.

### Prerequisites

Numpy, some graph libraries, sklearn, json and codecs.
needs python3 installed.


### Installing
Simply clone the repo, or Just copy past the MLP.py file. 
all the required code to train the model is in MLP.py. 


## Example
* trained MLP using XOR data-set, also performed gradient checking to see if the analytical gradient calculation is not buggy.
* trained MLP using Spiral data, to check if the model could learn non-linear function, plotted the Decision boundary to validate.
* trained MLP on IRIS dataset
* trained MLP on MNIST dataset (well, why not everyone is training on MNIST)


## Running the examples

Refer any of the example file to check the usage. 
I tried to keep the usage to other algorithm implementation in sklearn, by just needing to call the fit function by X and Y values to it.

PS: I moved the dataset to data/ in the process deleted XOR.data dataset, thus example breaks. but adding the copy of the xor dataset would remove the error.



## Authors

* **Ketan Kokane** - *Initial work* - (https://cs.rit.edu/~kk7471)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
