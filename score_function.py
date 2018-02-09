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

def score(weights):

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
    # Add weights corresponding to phi
    parser.add_weights(("phi W", 0), (pp, phiLayerSizes[0]))
    parser.add_weights(("phi bias", 0), phiLayerSizes[0])
    for ll in xrange(