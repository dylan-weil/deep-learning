import numpy as np
import pandas as pd
from scipy.special import softmax
import h5py
import matplotlib.pyplot as plt
from utils import *

np.random.seed(100)

def plot_sample_data(data, accuracy, predictions, true_values=None):
    fig, axes = plt.subplots(5, 2)
    plt.tight_layout()
    for i in range(10):
        reshaped_data = data[i].reshape((28, 28))
        prediction = predictions[i][0]
        if true_values is not None:
            true_value = true_values[i][0]

        axes[i%5, i//5].imshow(reshaped_data, cmap=plt.get_cmap('gray'))
        axes[i % 5, i // 5].set_title(f'Predicted: {prediction}, Actual: {true_value}')
    plt.suptitle(f'Accuracy: {accuracy}')
    plt.show()


class LogisticRegression:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.theta = np.zeros([output_size, input_size]) #Matrix of weights (can think of it like W in neural net

    def get_prediction_vector(self, input_data):
        softmax_input = np.dot(self.theta, input_data)
        return softmax(softmax_input)

    def predict(self, input_data):
        return np.argmax(self.get_prediction_vector(input_data))

    def gradient_of_loss(self, true_val, predicted_vals, input_data):
        loss_gradient = np.dot(neg_log_likelihood_deriv(true_val, predicted_vals), input_data.T)
        return loss_gradient

    def update_weights(self, learning_rate, loss_gradient):
        self.theta = self.theta - learning_rate * loss_gradient

    def sgd_train(self, input_data, output_data, steps, learning_rate_fcn, trace=False):
        data_features = input_data.shape[1] #Number of variables in each observation

        for step in range(steps):
            index = np.random.randint(0, len(input_data) - 1)
            x_data = input_data[index].reshape(data_features, 1)
            y_value = output_data[index]

            prediction_vector = self.get_prediction_vector(x_data)
            loss_grad = self.gradient_of_loss(y_value, prediction_vector, x_data)

            learning_rate = learning_rate_fcn(step)
            self.update_weights(learning_rate, loss_grad)

            if trace and step%1000 == 0:
                print(f'Step: {step}')

if __name__ == '__main__':

    with h5py.File('C:\\Users\\dylan\\OneDrive\\Documents\\Oxford\\HT 22\\Deep Learning\\Python\\MNISTdata.hdf5', 'r') as f:
        X_train = np.array(f['x_train'])
        Y_train = np.array(f['y_train'])
        Y_train = np.array([sublist[0] for sublist in Y_train])

        X_test = np.array(f['x_test'])
        Y_test = np.array(f['y_test'])
        Y_test = np.array([sublist[0] for sublist in Y_test])

    logit_model = LogisticRegression(X_train.shape[1], 10)
    logit_model.sgd_train(X_train, Y_train, 100000, continuous_learning_rate, trace=True)

    accuracy_counter = 0
    for i in range(len(X_test)):
        prediction = logit_model.predict(X_test[i])
        if prediction == Y_test[i]:
            accuracy_counter += 1

    print(accuracy_counter/len(X_test))