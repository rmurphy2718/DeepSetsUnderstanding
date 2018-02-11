# #######################################################
# Ryan Murphy | 2018
# Project: Understanding the Deep Sets paper through implementation.
#          Experiment 1: Rotated Gaussians.
# ---------------------------------------------------------------------
#
# Loss_and_Optim.py
#
# Calculate loss given score, and use a gradient
#   to minimize it.
#
# #######################################################

from score_function import *
import autograd.numpy as np

def loss_l2(target_vec, pred_vec):
    # We CANNOT compute the loss between a
    # (n,) and (n,p) array, even if the
    # (n,) array has p elements in second dim
    # The must BOTH be (n,p)
    assert target_vec.shape == pred_vec.shape
    return(np.sum((target_vec - pred_vec)**2))
#
# Loss function:
#  () weights must be the first parameter
#    since grad diffs wrt the first argument by default
#
#  () No need to pass data b/c it's built in to
#     the score function
def Loss(weights, targets, score_fun, loss_fun):
    preds = score_fun(weights)
    return(loss_fun(targets, preds))