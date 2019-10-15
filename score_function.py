# #######################################################
# Ryan Murphy | 2018
# Project: Understanding the Deep Sets paper through implementation.
#          Experiment 1: Rotated Gaussians.
# ---------------------------------------------------------------------
#
# score_function.py
#
# Run the score function: rho(sum(phi))
#   () Apply phi to every set element, which is follows a neural network
#         i.e. matmults and nonlinearities
#   > This gives representations, which we then:
#   () Sum
#   () Apply more layers.
# #######################################################
from WeightsParser import *
import autograd.numpy as np
import autograd.numpy.random as npr


def relu(x):
    return x * (x > 0) + 0.0


def sigmoid(x):
    return 0.5*(np.tanh(x/2.0) + 1)


def build_score_function(phi_layer_sizes, rho_layer_sizes, pp, activation=relu):
    """Return a function that forwards data through the model
    :param phi_layer_sizes: list giving sizes of hidden and output
    :param rho_layer_sizes: list giving sizes of rho layers
    :param pp: number of columns
    """
    assert isinstance(phi_layer_sizes, list)
    assert isinstance(rho_layer_sizes, list)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set up the weights parser needed for the task.
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    parser = WeightsParser()
    #
    # Add weights corresponding to phi
    #
    num_phi_layers = len(phi_layer_sizes)
    parser.add_weights(("phi W", 0), (pp, phi_layer_sizes[0]))
    parser.add_weights(("phi bias", 0), (1, phi_layer_sizes[0]))
    for ll in range(1, num_phi_layers):
        # Store number_of_columns for this layer and previous
        #   (previous #cols = current #rows)
        prev_cols = phi_layer_sizes[ll - 1]
        cur_cols = phi_layer_sizes[ll]
        parser.add_weights(("phi W", ll), (prev_cols, cur_cols))
        parser.add_weights(("phi bias", ll),  (1, cur_cols))
    #
    # Add weights corresponding to rho
    #
    # > Create a zip of rows & columns for each weight matrix
    # > The first number of rows will depend on the last phi
    num_rho_layers = len(rho_layer_sizes)
    row_col_tuples = zip([phi_layer_sizes[num_phi_layers - 1]] + rho_layer_sizes,
                         rho_layer_sizes + [1])  # Output is scalar: last matrix has only 1 column
    ll = 0
    for numRows, numCols in row_col_tuples:
        parser.add_weights(("rho W", ll), (numRows, numCols))
        parser.add_weights(("rho bias", ll), (1, numCols))
        ll += 1

    def score(weights, dat):
        """a function that forward data through the model"""
        assert isinstance(dat, np.ndarray)
        assert dat.ndim == 3
        #
        # Perform phi
        #
        W = parser.get(weights, ("phi W", 0))
        B = parser.get(weights, ("phi bias", 0))
        X = activation(np.matmul(dat, W) + B)
        for ll in range(1, num_phi_layers):
            W = parser.get(weights, ("phi W", ll))
            B = parser.get(weights, ("phi bias", ll))
            X = activation(np.matmul(X, W) + B)
        #
        # Add up the representations
        #
        rho_input = np.sum(X, axis = 1)
        #
        # Perform rho: another set of neural net
        #
        W = parser.get(weights, ("rho W", 0))
        B = parser.get(weights, ("rho bias", 0))
        X = activation(np.matmul(rho_input, W) + B)
        for ll in range(1, num_rho_layers + 1):  # +1 b/c there's an output layer.
            W = parser.get(weights, ("rho W", ll))
            B = parser.get(weights, ("rho bias", ll))
            # Do not apply activation to last layer
            if ll < num_rho_layers:
                X = activation(np.matmul(X, W) + B)
            else:
                return np.matmul(X, W) + B

    return score, parser


if __name__ == "__main__":
    # This can be used to check code before moving on
    score_fun, my_parser = build_score_function([1], [1], 16)
    # Print the representations of the objects
    print(score_fun)
    print(my_parser)

