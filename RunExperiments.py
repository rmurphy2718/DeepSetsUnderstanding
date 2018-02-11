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
N = 2**8
M = 500
inDat, targets = makeData(N, M, S, 1)

trainIndx = slice(0,  int(0.8*N))
validIndx = slice(int(0.8*N), int(0.9*N))
testIndx = slice(int(0.9*N), N+1)

trainData = inDat[trainIndx]
validData = inDat[validIndx]
testData = inDat[testIndx]

trainTargets = inDat[trainIndx]
validTargets = inDat[validIndx]
testTargets = inDat[testIndx]

#
# Set up network
#
phiLayers = [5, 5, 5] # hidden and output
rhoLayers = [7, 7]

score_fun, pars = buildScoreFunction(phiLayers, rhoLayers, inDat.shape[2])
Loss = buildLoss(score_fun, loss_l2)

def trainLoss(weights):
    return(Loss(weights, trainData, trainTargets))

grad_fun = grad(trainLoss)
# -------------------------------------------
# Implement a callback function
# -------------------------------------------
train_loss_curve = []
validation_loss_curve = []

#
# def callback(weights, iter):
#     if iter % 5 == 0:
#         # previously: train_preds = undo_norm(pred_fun(weights, train_smiles[:num_print_examples]))
#         # RLM change: added sigmoid to return probs
#         # > Get predictions
#         predictions = score_fun(weights, train_smiles[:num_print_examples])
#         pred_probs = rlm_sigmoid(predictions)
#         train_class_preds = get_class_preds(pred_probs)
#         #
#         del predictions
#         del pred_probs
#         # Loss
#         cur_loss = loss_fun(weights, train_smiles[:num_print_examples], train_targets[:num_print_examples])
#         training_curve.append(cur_loss)
#         #
#         cur_train_acc = acc(train_class_preds, train_raw_targets[:num_print_examples])
#         acc_train.append(cur_train_acc)
#         #
#         print("~~~")
#         print "Iteration", iter, "\nTrain Loss: ", cur_loss, \
#             "\nTrain Accuracy: ", cur_train_acc
#         #
#         if validation_smiles is not None:
#             predictions = pred_fun(weights, validation_smiles)
#             pred_probs = rlm_sigmoid(predictions)
#             validation_class_preds = get_class_preds(pred_probs)
#             validation_loss = loss_fun(weights, validation_smiles[:num_print_examples],
#                                        validation_raw_targets[:num_print_examples])
#             validation_curve.append(validation_loss)
#             cur_val_acc = acc(validation_class_preds, validation_raw_targets)
#             acc_validation.append(cur_val_acc)
#             print "Validation Loss:", validation_loss
#             print "Validation Accuracy: ", cur_val_acc
#         #
#         print "max of weights", np.max(np.abs(weights))
#         print("~~~")

# ------------------------------------------
# Train it
#
# ------------------------------------------
init_weights = 0.25 * npr.randn(len(pars))
adam(grad_fun, init_weights, 5)
