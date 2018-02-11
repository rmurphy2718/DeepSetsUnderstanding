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
import matplotlib.pyplot as plt
#  Make dataset
S = np.array((10, 2, 2, 7)).reshape(2, 2)
N = 2**9
M = 500
inDat, targets = makeData(N, M, S, 1)

trainIndx = slice(0,  int(0.8*N))
validIndx = slice(int(0.8*N), int(0.9*N))
testIndx = slice(int(0.9*N), N+1)

trainData = inDat[trainIndx]
validData = inDat[validIndx]
testData = inDat[testIndx]

trainTargets = targets[trainIndx]
validTargets = targets[validIndx]
testTargets = targets[testIndx]

#
# Set up network
#
phiLayers = [5, 5, 5] # hidden and output
rhoLayers = [7, 7]

score_fun, pars = buildScoreFunction(phiLayers, rhoLayers, trainData.shape[2])
Loss = buildLoss(score_fun, loss_l2)

def trainLoss(weights):
    return(Loss(weights, trainData, trainTargets))


grad_fun = grad(trainLoss)
# -------------------------------------------
# Implement a callback function
# -------------------------------------------
train_MSE_curve = []
valid_MSE_curve = []

def callback(weights, iter):
    if iter % 5 == 0:
        curr_train_MSE = trainLoss(weights)/float(len(trainTargets))
        train_MSE_curve.append(curr_train_MSE )
        #
        curr_valid_MSE = Loss(weights, validData, validTargets)/float(len(validTargets))
        valid_MSE_curve.append(curr_valid_MSE )
        #
        print("~~~")
        print "Iteration", iter, "\nTrain MSE: ", curr_train_MSE, \
            "\nValidation MSE: ", curr_valid_MSE
        #
        #
        print "\nmax of weights", np.max(np.abs(weights))
        print("~~~")

# ------------------------------------------
# Train it
#
# ------------------------------------------
init_weights = 0.25 * npr.randn(len(pars))
trained_weights = adam(grad_fun, init_weights, 250, callback)
# ------------------------------------------
# Loss plots
#
# ------------------------------------------
# Plotting code:
#  https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
plt.plot(train_MSE_curve) # it's loss, but they call it training curve.
plt.plot(valid_MSE_curve)
plt.title('Loss plots')
plt.ylabel('MSE')
plt.xlabel('Iteration / Callback Frequency')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

#
# Calculate loss
#
SSE = Loss(trained_weights, testData, testTargets)
MSE = SSE/float(len(testTargets))
print("Mean Square Error: %f " % MSE)

# permute
testIndx_New = np.arange(0, len(testTargets))
npr.shuffle(testIndx_New)
testData_New = testData[testIndx_New,:,:]

SSE_1 = Loss(trained_weights, testData, testTargets)
MSE_1 = SSE_1/float(len(testTargets))
print("Mean Square Error after permutation: %f " % MSE)


# ======================================
#
# Compare to performance achievable by "cheating"
# Use knowledge that it's from a Gaussian
# Compute Variance
# Call formula for entropy from variance
# ======================================
cheatPreds = []
for jj in xrange(0,testData.shape[0]):
    Y1 = testData[jj, :, 0]
    s2 = np.var(Y1)
    cheatPreds.append(getGaussianEntropy(s2))

cheatPreds = np.array(cheatPreds).reshape(len(cheatPreds), 1)

loss_l2(testTargets, cheatPreds)/len(cheatPreds)


