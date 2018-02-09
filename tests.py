# Test the variance formula.
x = []
y = []
alp = pi/8.0
R = np.array([[cos(alp),-sin(alp)],[sin(alp),cos(alp)]])
SIG = np.array([[4,0.5],[0.5,3]])
for ii in xrange(0,20000):
    x.append(npr.multivariate_normal([0,0], SIG, 1).T)
    y.append(np.matmul(R, x[ii]))


y1 = [float(piece1) for (piece1, piece2) in y]
print(np.var(y1))

alph = alp
ans = SIG[0,0]*(cos(alph)**2) - \
                      2 * SIG[0, 1] * cos(alph) * sin(alph) + \
                      SIG[1, 1] * (sin(alph) ** 2)
print(ans)



#
#
#
test = makeData(5, 200, np.array([[3,1],[1,4]]))
test1 = test[0,:,:]
np.mean(test1, axis = 0)