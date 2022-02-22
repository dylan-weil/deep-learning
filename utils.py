import numpy as np

def continuous_learning_rate(step_number, c_0=1, c_1=1, c_2=1):
    return c_0/(c_2 * step_number + c_1)


def piecewise_learning_rate(step_number, c_0=10, c_1=10):
    return c_0**-np.floor(np.log10(step_number + 1)/np.log10(c_1))
    #The value of c_1 is the base of the log, via the change of base formula for logs


def sigmoid(z):
    return_val = 1/(1 + np.exp(-z))
    if np.isnan(return_val).any():
        raise ValueError('Invalid Return Value')
    else:
        return return_val


def sigmoid_deriv(z):
    return sigmoid(z) * (1 - sigmoid(z))


def identity(z):
    return z


def identity_deriv(z):
    return 1


def neg_log_likelihood(actual, predictions):
    return -np.log(predictions[actual])


def neg_log_likelihood_deriv(onehot_true_vals, predictions):
    # Transformation should be a pre-fitted onehot_encoder
    # return -(np.eye(1, len(predictions), onehot_true_vals).T - predictions)
    return -(onehot_true_vals - predictions)

def convolution(matrix, filter, pad=0, stride=1):
    filter_rows = filter.shape[0]
    filter_cols = filter.shape[1]

    input_rows = matrix.shape[0]
    input_cols = matrix.shape[1]

    if pad != 0:
        padded_matrix = np.zeros(shape=[input_rows + 2*pad, input_cols + 2*pad])
        padded_matrix[pad:-pad, pad:-pad] = matrix
    else:
        padded_matrix = matrix

    output_rows = int(np.floor((input_rows - filter_rows + 2*pad)/stride) + 1)
    output_cols = int(np.floor((input_cols - filter_cols + 2*pad)/stride) + 1)

    output_matrix = np.zeros(shape=[output_rows, output_cols])

    for i in range(output_rows):
        strided_i = i * stride
        for j in range(output_cols):
            strided_j = j * stride

            input_submatrix = padded_matrix[strided_i:filter_rows+strided_i, strided_j:filter_cols+strided_j]
            convolution_terms = input_submatrix * filter
            output_matrix[i, j] = convolution_terms.sum()

    return output_matrix

# matrix = np.array([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]])
# filter = np.array([[4, 6], [8, 10]])
#
# print(convolution(matrix, filter, pad=1, stride=3))
