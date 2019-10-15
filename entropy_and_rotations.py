# #######################################################
# Ryan Murphy | 2018
# Project: Understanding the Deep Sets paper through implementation.
#          Experiment 1: Rotated Gaussians.
# ---------------------------------------------------------------------
#  helper functions to calculate analytical entropy
# ######################################################

from math import log, pi, cos, sin

def getGaussianEntropy(sigsq):
    """
    Analytic formula for entropy of a univariate Gaussian based on the variance.
    :param sigsq: variance
    """
    return 0.5*log(2.0*pi*sigsq) + 0.5


def getEntropyRotated(alph, SIG):
    """
      Return entropy of first component of a bivariate Gaussian rotated by alpha via rotation R(alpha)
      No rotations actually necessary -- we have closed-form formulae for what would happen if we did rotate
    """
    # SIG is sigma matrix of the original data
    assert SIG[0, 1] == SIG[1, 0], "Matrix SIG should be symmetric."
    # Calculate variance first component under rotated Gaussian
    rotatedVariance = SIG[0, 0]*(cos(alph)**2) - \
                      2 * SIG[0, 1] * cos(alph) * sin(alph) + \
                      SIG[1, 1] * (sin(alph) ** 2)
    # Return entropy based on new variance
    getGaussianEntropy(rotatedVariance)


