import numpy as np
from utils import *

class Layer:
    def __init__(self, input_size, output_size, activation_fcn, activation_fcn_deriv, label=None):
        self.label = label
        self.activation_fcn = activation_fcn
        self.activation_fcn_deriv = activation_fcn_deriv

        self.weights = np.random.normal(0, 1, [output_size, input_size])
        self.bias = np.random.normal(0, 1, [output_size]).reshape(output_size, 1) #Reshaping into a column vector (AKA 1 col matrix)

        #Set these when the layer is activated
        self.inputs = None
        self.activation_inputs = None
        self.outputs = None

        #Set these during backpropagation step, wipe them after weight update
        self.loss_d_weights = None
        self.loss_d_inputs = None
        self.loss_d_bias = None
        self.loss_d_activation_inputs = None
        self.loss_d_outputs = None

    def activate(self, input_data):
        self.inputs = input_data
        self.activation_inputs = np.dot(self.weights, input_data) + self.bias
        self.outputs = self.activation_fcn(self.activation_inputs)
        return self.outputs

    def reset_derivatives(self):
        self.loss_d_weights = None
        self.loss_d_inputs = None
        self.loss_d_bias = None
        self.loss_d_activation_inputs = None
        self.loss_d_outputs = None

    def set_backprop_derivatives(self, loss_d_outputs, batch_size):
        self.loss_d_outputs = loss_d_outputs #rho_d_H or rho_d_U, remember that the output loss of one layer is the input loss of the next layer

        delta_z_elementwise_prod = self.loss_d_outputs * self.activation_fcn_deriv(self.activation_inputs)

        self.loss_d_weights = np.dot(delta_z_elementwise_prod, self.inputs.T) # rho_d_W
        self.loss_d_bias = delta_z_elementwise_prod # rho_d_b

        self.loss_d_inputs = np.dot(self.weights.T, self.loss_d_outputs)  # rho_d_x

class ConvolutionLayer(Layer):
    def __init__(self, input_size, output_size, activation_fcn, activation_fcn_deriv, filters, label=None):
        super().__init__(input_size, output_size, activation_fcn, activation_fcn_deriv, label=label)

        self.filters = filters
        self.loss_d_filter = None
        #Remember that the output size might be determined by the input size and the filter

    def reset_derivatives(self):
        super().reset_derivatives()
        self.loss_d_filter = None

#initial commit comment
