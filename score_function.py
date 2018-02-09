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

def score(weights):

def buildScoreFunction(phiLayers, rhoLayers, dat):
    assert isinstance(phiLayers, list)
    assert isinstance(rhoLayers, list)
    #
    parser = WeightsParser()
    # Add weights corresponding to phi
    for ll in xrange()
