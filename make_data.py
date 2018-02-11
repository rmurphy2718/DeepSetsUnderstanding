# #######################################################
# Ryan Murphy | 2018
# Project: Understanding the Deep Sets paper through implementation.
#          Experiment 1: Rotated Gaussians.
# ---------------------------------------------------------------------
# make_data.py:
#   () Make a full dataset for this task, containing both training and test
#   () The data each represents different data SETS of bivariate data of size M, each drawn from bivariate normals with mean 0 and some variance
#       > The variances are determined through a rotation.
#   () The goal is: Given a new SET of bivariate data, predict the entropy of that set.
#
#  Approach: will store each set as a matrix, and
#   the whole data is a tensor.
# #######################################################
import autograd.numpy as np
import autograd.numpy.random as npr
from math import pi, cos, sin
from entropy_and_rotations import *
def makeData(N, M, SIG, seed=None):
    # Set seed if desired
    if seed is not None:
        npr.seed(seed)
    # Initialize data
    dat = np.zeros((N, M, 2)) # N tensors, M rows, 2 columns
    targs = np.zeros(N).reshape(N,1)
    for ii in xrange(0,N):
        # Draw a random rotation angle
        alp = np.random.uniform(0, pi)
        # Construct rotation matrix
        R = np.array([[cos(alp),-sin(alp)],[sin(alp),cos(alp)]])
        # Get variance matrix of new data
        #   We suppose Var(X) = SIGMA, and we compute RX
        #   hence variance is R SIGMA R^T
        VarMat = np.matmul(np.matmul(R, SIG), R.T)
        #
        # Draw bivariate normal
        dat[ii, :, :] = npr.multivariate_normal([0, 0], VarMat, M)
        #
        # Target is entropy of first component
        targs[ii] = getGaussianEntropy(VarMat[0,0])
    #
    return(dat,targs)
