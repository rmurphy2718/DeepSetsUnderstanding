# #######################################################
# Ryan Murphy | 2018
# Project: Understanding the Deep Sets paper through implementation.
#          Experiment 1: Rotated Gaussians.
# ---------------------------------------------------------------------
#
# score_function.py
#
# Run the score function: rho(sum(phi))
#   () Apply phi to every set element, which is follows a "neural network"
#         i.e. matmults and nonlinearities
#   > This gives representations
#   () Sum them
#   () Apply more nonlinearities.
# #######################################################
from WeightsParser import *
import autograd.numpy as np
import autograd.numpy.random as npr

def relu(x):
    return x * (x > 0) + 0.0

def sigmoid(x):
    return 0.5*(np.tanh(x/2.0) + 1)

#def score(weights):

def buildScoreFunction(phiLayerSizes, rhoLayerSizes, dat, activation=relu):
    assert isinstance(phiLayerSizes, list)
    assert isinstance(rhoLayerSizes, list)
    assert isinstance(dat, np.ndarray)
    assert dat.ndim == 3
    # Get number of columns in data
    pp = dat.shape[2] # shape = (num matrices, num rows, num columns)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set up the weights parser needed for the task.
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    parser = WeightsParser()
    #
    # Add weights corresponding to phi
    #
    numPhiLayers = len(phiLayerSizes)
    parser.add_weights(("phi W", 0), (pp, phiLayerSizes[0]))
    parser.add_weights(("phi bias", 0), (1, phiLayerSizes[0]))
    for ll in xrange(1, numPhiLayers):
        # Store number_of_columns for this layer and previous
        #   (previous #cols = current #rows)
        prevCols = phiLayerSizes[ll - 1]
        curCols = phiLayerSizes[ll]
        parser.add_weights(("phi W", ll), (prevCols, curCols))
        parser.add_weights(("phi bias", ll),  (1, curCols))
    #
    # Add weights corresponding to rho
    #
    # > Create a zip of rows & columns for each weight matrix
    # > The first number of rows will depend on the last phi
    numRhoLayers = len(rhoLayerSizes)
    row_col_tuples = zip([phiLayerSizes[numPhiLayers-1]] + rhoLayerSizes,
                           rhoLayerSizes + [1]) # Output is scalar: last matrix has only 1 column
    ll = 0
    for numRows, numCols in row_col_tuples:
        parser.add_weights(("rho W", ll), (numRows, numCols))
        parser.add_weights(("rho bias", ll), (1, numCols))
        ll += 1
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build up the score function
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def score(weights):
        #
        # Perform phi
        #
        W = parser.get(weights, ("phi W", 0))
        B = parser.get(weights, ("phi bias", 0))
        X = activation(np.matmul(dat, W) + B)
        for ll in xrange(1, numPhiLayers):
            W = parser.get(weights, ("phi W", ll))
            B = parser.get(weights, ("phi bias", ll))
            X = activation(np.matmul(X, W) + B)
        #
        # Add up the representations
        #
        RhoInput = np.sum(X, axis = 0)
        #
        # Perform rho: another set of neural net
        #
        W = parser.get(weights, ("rho W", 0))
        B = parser.get(weights, ("rho bias", 0))
        X = activation(np.matmul(RhoInput, W) + B)
        for ll in xrange(1, numRhoLayers + 1): # +1 b/c there's an output layer.
            W = parser.get(weights, ("rho W", ll))
            B = parser.get(weights, ("rho bias", ll))
            # Do not apply activation to last layer
            if ll < numRhoLayers:
                X = activation(np.matmul(X, W) + B)
            else:
                return np.matmul(X, W) + B
    #

    #
    #
    return score, parser
