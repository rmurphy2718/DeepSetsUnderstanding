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
#
# Adam optimizer
# Not my function!
#   Taken directly from Duvenaud and MacLaurin's fingerprint code.
#

def adam(grad, x, callback=None, num_iters=100,
         step_size=0.001, b1=0.9, b2=0.999, eps=10**-8):
    """Adam as described in http://arxiv.org/pdf/1412.6980.pdf.
    It's basically RMSprop with momentum and some correction terms."""
    m = np.zeros(len(x))
    v = np.zeros(len(x))
    for i in range(num_iters):
        print("Iteration " + str(i))
        sys.stdout.flush()
        g = grad(x, i)
        if callback: callback(x, i)
        m = (1 - b1) * g      + b1 * m  # First  moment estimate.
        v = (1 - b2) * (g**2) + b2 * v  # Second moment estimate.
        mhat = m / (1 - b1**(i + 1))    # Bias correction.
        vhat = v / (1 - b2**(i + 1))
        x -= step_size*mhat/(np.sqrt(vhat) + eps)
    #
    print("Finished Adam.")
    return x
