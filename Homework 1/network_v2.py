import numpy as np
from layer import Layer
from scipy.special import softmax
from utils import *
from sklearn.preprocessing import OneHotEncoder

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
        return softmax(input_data, axis=0)

    def backprop(self, onehot_true_values, prediction_vector, batch_size):
        rho_d_U = neg_log_likelihood_deriv(onehot_true_values, prediction_vector)
        output_error = rho_d_U
        for layer in reversed(self.layers):
            layer.set_backprop_derivatives(output_error, batch_size)
            output_error = layer.loss_d_inputs

    def update_parameters(self, learning_rate, batch_size):
        for layer in self.layers:
            layer.weights -= learning_rate * layer.loss_d_weights
            layer.bias -= learning_rate * layer.loss_d_bias[:, 0].reshape(-1, 1)
            layer.reset_derivatives()

    def predict(self, input_data):
        return np.argmax(self.feed_forward(input_data))

    def sgd_train(self, input_data, output_data, steps, learning_rate_fcn, batch_size=1, trace=False):
        onehot_encoder = OneHotEncoder(sparse=False)
        onehot_encoder.fit(output_data.reshape(-1, 1))

        data_features = input_data.shape[1]  # Number of variables in each observation
        correct_predictions = 0

        for step in range(steps):
            index = np.random.randint(0, len(input_data) - 1, size=batch_size)
            x_data = input_data[index].reshape(data_features, batch_size)
            y_values = output_data[index]

            prediction_vector = self.feed_forward(x_data) #Make sure to properly include softmax in backprop

            onehot_y_values = onehot_encoder.transform(y_values.reshape(-1, 1)).T
            self.backprop(onehot_y_values, prediction_vector, batch_size)

            learning_rate = learning_rate_fcn(step)
            self.update_parameters(learning_rate, batch_size)

            if trace:
                correct_predictions += sum(self.predict(x_data) == y_values)
                if step % 1000 == 0:
                    print(f'Step: {step}, Accuracy: {correct_predictions/1000}')
                    correct_predictions = 0