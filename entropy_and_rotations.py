# #######################################################
# Ryan Murphy | 2018
# Project: Understanding the Deep Sets paper through implementation.
#          Experiment 1: Rotated Gaussians.
# ---------------------------------------------------------------------
#  entropy_and_rotations.py contains
#  helper functions to calculate entropy
# ######################################################

from math import log, pi, cos, sin
import autograd.numpy as np
import autograd.numpy.random as npr

# Return entropy of a univariate Gaussian based on the variance.
# Analytic formula
def getGaussianEntropy(sigsq):
    return 0.5*log(2.0*pi*sigsq) + 0.5

# Return entropy of first component of a bivariate Gaussian rotated by alpha via rotation R(alpha)
#  No rotations actually nescessary -- we have closed-form formulae for what would happen if we did rotate
def getEntropyRotated(alph, SIG):
    # SIG is matrix of original data
    assert SIG[0, 1] == SIG[1, 0], "Matrix SIG should be symmetric."
    # Calculate variance first component under rotated Gaussian
    rotatedVariance = SIG[0, 0]*(cos(alph)**2) - \
                      2 * SIG[0, 1] * cos(alph) * sin(alph) + \
                      SIG[1, 1] * (sin(alph) ** 2)
    # Return entropy based on new variance
    getGaussianEntropy(rotatedVariance)


