import numpy as np
import pandas as pd
from utils import *
import h5py
from layer import Layer
from network_v2 import Network

np.random.seed(10000)

if __name__ == '__main__':
    network = Network()
    network.add_layer(Layer(784, 64, sigmoid, sigmoid_deriv, label='Hidden Layer 1'))
    network.add_layer(Layer(64, 10, identity, identity_deriv, label='Output Layer'))

    with h5py.File('C:\\Users\\dylan\\OneDrive\\Documents\\Oxford\\HT 22\\Deep Learning\\Python\\MNISTdata.hdf5','r') as f:
        X_train = np.array(f['x_train'])
        Y_train = np.array(f['y_train'])
        Y_train = np.array([sublist[0] for sublist in Y_train])

        X_test = np.array(f['x_test'])
        Y_test = np.array(f['y_test'])
        Y_test = np.array([sublist[0] for sublist in Y_test])

    learning_rate_fcn = lambda x: continuous_learning_rate(x, .1, 1, 0) #Constant learning rate seems to perform way better
    network.sgd_train(X_train, Y_train, 100000, learning_rate_fcn, batch_size=1, trace=True)

    accuracy_counter = 0
    for i in range(len(X_test)):
        x_test_sample = X_test[i].reshape(X_test.shape[1], 1)
        y_test_sample = Y_test[i]

        prediction = network.predict(x_test_sample)
        print(prediction, Y_test[i])
        if prediction == Y_test[i]:
            accuracy_counter += 1

    print(accuracy_counter/len(X_test))

