import autograd.numpy as np
import autograd.numpy.random as npr
from score_function import *
from Loss_and_Optim import *
from autograd import grad

# # Test the variance formula
# x = []
# y = []
# alp = pi/8.0
# R = np.array([[cos(alp),-sin(alp)],[sin(alp),cos(alp)]])
# SIG = np.array([[4,0.5],[0.5,3]])
# for ii in xrange(0,20000):
#     x.append(npr.multivariate_normal([0,0], SIG, 1).T)
#     y.append(np.matmul(R, x[ii]))
#
#
# y1 = [float(piece1) for (piece1, piece2) in y]
# print(np.var(y1))
#
# alph = alp
# ans = SIG[0,0]*(cos(alph)**2) - \
#                       2 * SIG[0, 1] * cos(alph) * sin(alph) + \
#                       SIG[1, 1] * (sin(alph) ** 2)
# print(ans)
#
#
#
# #
# #
# #
# test = makeData(5, 200, np.array([[3,1],[1,4]]))
# test1 = test[0,:,:]
# np.mean(test1, axis = 0)
#
# #
# # 3-D array: matmults with 2-D array by layer
# #
# #
# #
# # > Set up matrices
A = np.arange(4,8).reshape(2,2)
B = np.arange(-3, 1).reshape(2,2)
C = np.array([-1, 6, 2, 10]).reshape(2,2)
Z = np.array([3, 2, -5, 9]).reshape(2,2)
# #
# # Make A and B a 3-D array
tens = np.stack([A, B, C], axis = 0)
# #
# # Matrix multiply each layer in tensor with Z
# #
# print(np.matmul(tens, Z))
# # Compare with the direct multiplications AZ, BZ, CZ.
# print(np.matmul(A, Z))
# print(np.matmul(B, Z))
# print(np.matmul(C, Z))
# # It works!
#
# #
# # Test matrix multiply plus vector...
# #   How does the vector get coerced to a matrix?
# #
# #
# B = np.arange(-3,1).reshape(2, 2)
# A = np.arange(1,5).reshape(2, 2)
# b = np.array([30,1]).reshape(1, 2)
#
# np.matmul(A, B) + b
# # Good! The first element gets stretched down the first column
# # the second element gets stretched down the second column, etc.
#
#
# relu(A)
# relu(C)
#
# A = npr.randn(4).reshape(2,2)
# relu(A)
S = np.array((10, 2, 2, 7)).reshape(2,2)
tIn, tOut = makeData(3, 500, S, 1)

y1 = tIn[1,:,0]
getGaussianEntropy(np.var(y1))
tOut[1]
#
# Syntax check buildScoreFunction
#
#
sco, pars = buildScoreFunction([5,6,7],
                          [4,5,3],
                          tens,
                          relu)
# Weight parser has correct number of weights:

len(pars)
(
2*5 + 5 +
5*6 + 6 +
6*7 + 7 +
7*4 + 4 +
4*5 + 5 +
5*3 + 3 +
3*1 + 1
)
#
init_weights = .25 * npr.randn(len(pars))
pars.get(init_weights, ("rho W", 3))

print(sco(init_weights))

testo = sco(init_weights, tens)
type(testo)
testo.shape


targets = np.array((0.1,0.6)).reshape(2,1)
Loss = buildLoss(targets, sco, loss_l2)
Loss(init_weights)

(testo[0]-0.1)**2 + (testo[1]-0.6)**2

#
# Gradient
#
grad_func = grad(Loss)
test_w_grad = grad_func(init_weights, targets, sco, loss_l2)



tens2 = np.arange(0,2*3*4).reshape(2,3,4)
print(tens2)
print(np.sum(tens2, axis = 1))