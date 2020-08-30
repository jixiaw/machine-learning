import numpy as np

def sigmiod(z):
    return 1 / (1 + np.exp(-z))


def logistic(X, y):
    '''
    LR Logistic Regression.

    INPUT:  X: training sample features, P-by-N matrix.
            y: training sample labels, 1-by-N row vector.

    OUTPUT: w: learned parameters, (P+1)-by-1 column vector.
    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    # YOUR CODE HERE
     # begin answer
    Y = np.array(y)
    Y[Y < 0] = 0
    x = np.vstack((np.ones((1, N)), X))
    lr = 0.1
    steps = 100000
    lw = 0
    for step in range(steps):
        pred = sigmiod(np.dot(w.T, x))
        grad = np.dot(x, (pred - Y).T)
        w = w - grad * lr
        lw_new = -np.sum(Y * pred + (1 - Y) * (1 - pred))
        # if step % 100 == 0:
        #     print("step:", step, "L(w): ", lw_new)
        if abs(lw_new - lw) < 1e-3:
            break
        lw = lw_new
    # end answer
    
    return w
