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
import numpy as np

#def score(weights):

def buildScoreFunction(phiLayerSizes, rhoLayerSizes, dat):
    assert isinstance(phiLayerSizes, list)
    assert isinstance(rhoLayerSizes, list)
    assert isinstance(dat, np.ndarray)
    assert dat.ndim == 3
    # Get number of columns in data
    pp = dat.shape[2]
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
    # Perform phi function
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


