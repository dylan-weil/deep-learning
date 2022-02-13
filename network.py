import numpy as np
from layer import Layer
from scipy.special import softmax
from utils import *

class Network:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer: Layer):
        self.layers.append(layer)

    def clear_layer_derivatives(self):
        for layer in self.layers:
            layer.loss_d_weights = None
            layer.loss_d_inputs = None
            layer.loss_d_bias = None
            layer.loss_d_activation_inputs = None
            layer.loss_d_outputs = None

    def feed_forward(self, input_data):
        for layer in self.layers:
            input_data = layer.activate(input_data)
        return softmax(input_data)

    def backprop(self, true_value, prediction_vector):
        final_layer = self.layers[-1]
        rho_d_U = neg_log_likelihood_deriv(true_value, prediction_vector)

        #To do this in the layers you would
        final_layer.loss_d_outputs = rho_d_U #This implicitly assumes that the softmax is part of the final layer, not something that happens separately
        final_layer.loss_d_weights = np.dot(final_layer.loss_d_outputs, final_layer.inputs.T)
        final_layer.loss_d_bias = final_layer.loss_d_outputs
        final_layer.loss_d_inputs = np.dot(final_layer.weights.T, final_layer.loss_d_outputs)#This is delta in the one hidden layer case, since input to final layer is H

        for layer_index in range(len(self.layers) - 2, -1, -1):
            layer = self.layers[layer_index]
            next_layer = self.layers[layer_index+1]

            delta_z_elementwise_prod = next_layer.loss_d_inputs * layer.activation_fcn_deriv(layer.activation_inputs)

            layer.loss_d_outputs = next_layer.loss_d_inputs #rho_d_H
            layer.loss_d_weights = np.dot(delta_z_elementwise_prod, layer.inputs.T) #rho_d_W
            layer.loss_d_bias = delta_z_elementwise_prod #rho_d_b1
            layer.loss_d_inputs = np.dot(layer.weights.T, layer.loss_d_outputs) #rho_d_x

    def update_parameters(self, learning_rate):
        for layer in self.layers:
            layer.weights -= learning_rate * layer.loss_d_weights
            layer.bias -= learning_rate * layer.loss_d_bias
        self.clear_layer_derivatives()

    def predict(self, input_data):
        return np.argmax(self.feed_forward(input_data))

    def sgd_train(self, input_data, output_data, steps, learning_rate_fcn, trace=False):
        data_features = input_data.shape[1]  # Number of variables in each observation
        correct_predictions = 0

        for step in range(steps):
            index = np.random.randint(0, len(input_data) - 1)
            x_data = input_data[index].reshape(data_features, 1)
            y_value = output_data[index]

            prediction_vector = self.feed_forward(x_data) #Make sure to properly include softmax in backprop
            self.backprop(y_value, prediction_vector)

            learning_rate = learning_rate_fcn(step)
            self.update_parameters(learning_rate)

            if trace:
                correct_predictions += self.predict(x_data) == y_value
                if step % 1000 == 0:
                    print(f'Step: {step}, Accuracy: {correct_predictions/1000}')
                    correct_predictions = 0