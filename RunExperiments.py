# #######################################################
# Ryan Murphy | 2018
# Project: Understanding the Deep Sets paper through implementation.
#          Experiment 1: Rotated Gaussians.
# ---------------------------------------------------------------------
#
# Load in the data and train neural network.
#
#
############################################################
from make_data import *
from Loss_and_Optim import *
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
#  Make dataset
S = np.array((10, 2, 2, 7)).reshape(2, 2)
inDat, targets = makeData(2**8, 500, S, 1)
#
# Set up network
#
phiLayers = [5, 5, 5]
rhoLayers = [7, 7, 7]

score_fun, pars = buildScoreFunction(phiLayers, rhoLayers, inDat)
Loss = buildLoss(targets, score_fun, loss_l2)

grad_fun = grad(Loss)
#
# Train it
#
init_weights = 0.25 * npr.randn(len(pars))
adam(grad_fun, init_weights, 5)
