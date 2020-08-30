import numpy as np
def sigmiod(z):
    return 1 / (1 + np.exp(-z))

def logistic_r(X, y, lmbda):
    '''
    LR Logistic Regression.

      INPUT:  X:   training sample features, P-by-N matrix.
              y:   training sample labels, 1-by-N row vector.
              lmbda: regularization parameter.

      OUTPUT: w    : learned parameters, (P+1)-by-1 column vector.
    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    # YOUR CODE HERE
    # begin answer
    Y = np.array(y)
    Y[Y < 0] = 0
    x = np.vstack((np.ones((1, N)), X))
    lr = 0.1
    steps = 500
    lw = 0
    for step in range(steps):
        pred = sigmiod(np.dot(w.T, x))
        grad = np.dot(x, (pred - Y).T) + lmbda * w
        grad[0, 0] = grad[0, 0] - lmbda * w[0, 0]
        grad = grad / N
        w = w - grad * lr
        lw_new = -np.sum(Y * pred + (1 - Y) * (1 - pred)) + 1/2 * np.dot(w.T, w)[0, 0]
#         print(lmbda, step, lw_new)
        # if step % 100 == 0:
        #     print("step:", step, "L(w): ", lw_new)
        if abs(lw_new - lw) < 1e-3:
            break
        lw = lw_new
    # end answer
    return w
