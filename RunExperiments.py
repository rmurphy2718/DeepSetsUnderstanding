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


def callback(weights, iteration):
    """A callback function used for reporting during training"""
    if iteration % 5 == 0:
        curr_train_MSE = train_loss(weights) / float(len(train_targets))
        train_MSE_curve.append(curr_train_MSE)

        curr_valid_MSE = Loss(weights, valid_data, valid_targets) / float(len(valid_targets))
        valid_MSE_curve.append(curr_valid_MSE )

        print("~"*3)
        print(f"Iteration {iteration}\nTrain MSE: {curr_train_MSE}\nValidation MSE: {curr_valid_MSE}")
        print(f"Maximum of abs(weights): {np.max(np.abs(weights))}")
        print("~"*3)


if __name__ == "__main__":
    #
    #  Make dataset
    #
    S = np.array((10, 2, 2, 7)).reshape(2, 2)
    N = 2**9
    M = 500
    in_dat, targets = make_data(N, M, S, 1)
    #
    # Split into train, test, val
    #
    train_idx = slice(0, int(0.8 * N))
    valid_idx = slice(int(0.8 * N), int(0.9 * N))
    test_idx = slice(int(0.9 * N), N + 1)

    train_data = in_dat[train_idx]
    valid_data = in_dat[valid_idx]
    test_data = in_dat[test_idx]

    train_targets = targets[train_idx]
    valid_targets = targets[valid_idx]
    test_targets = targets[test_idx]
    #
    # Set up network, hard-coded for simplicity
    #
    phi_layers = [5, 5, 5]  # hidden and output
    rho_layers = [7, 7]

    score_fun, pars = build_score_function(phi_layers, rho_layers, train_data.shape[2])
    Loss = buildLoss(score_fun, loss_l2)

    def train_loss(weights):
        return Loss(weights, train_data, train_targets)

    grad_fun = grad(train_loss)
    #
    # Train model
    #
    train_MSE_curve = []
    valid_MSE_curve = []

    init_weights = 0.25 * npr.randn(len(pars))
    trained_weights = adam(grad_fun, init_weights, 250, callback)
    #
    # Plots to monitor training
    # Plotting code:
    #  https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
    #
    # Will open up in a window
    #
    plt.plot(train_MSE_curve)
    plt.plot(valid_MSE_curve)
    plt.title('Loss plots')
    plt.ylabel('MSE')
    plt.xlabel('Iteration / Callback Frequency')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()  # Open in window
    #
    # Calculate loss
    #
    SSE = Loss(trained_weights, test_data, test_targets)
    MSE = SSE/float(len(test_targets))
    print("Mean Square Error: %f " % MSE)
    #
    # Sanity check of permutation-invariance of implemented model
    #
    # permute
    test_idx_new = np.arange(0, len(test_targets))
    npr.shuffle(test_idx_new)
    test_data_new = test_data[test_idx_new, :, :]

    SSE_1 = Loss(trained_weights, test_data, test_targets)
    MSE_1 = SSE_1/float(len(test_targets))
    print("Mean Square Error after permutation: %f " % MSE)
    # ======================================
    # Compare to performance achievable by "cheating"
    # Use knowledge that it's from a Gaussian
    # Compute Variance
    # Call formula for entropy from variance
    # ======================================
    cheat_preds = []
    for jj in range(0, test_data.shape[0]):
        Y1 = test_data[jj, :, 0]
        s2 = np.var(Y1)
        cheat_preds.append(getGaussianEntropy(s2))

    cheat_preds = np.array(cheat_preds).reshape(len(cheat_preds), 1)

    cheat_loss = loss_l2(test_targets, cheat_preds) / len(cheat_preds)
    print(f"Loss using exact formula: {cheat_loss}")


