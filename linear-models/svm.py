import numpy as np
from scipy.optimize import fmin_tnc

def func(w, *args):
    x, y, lr = args
    p = np.dot(w, x)  
    idx = np.where(p * y[0] < 1)[0]  # 分错的点
    err = p[idx] - y[0, idx]  
    cost = (np.dot(err, err.T) + lr * np.dot(w, w)) / 2 # 损失函数  
    grad = np.dot(x[:, idx], err) + lr * w # 梯度
    return cost, grad

def svm(X, y):
    '''
    SVM Support vector machine.

    INPUT:  X: training sample features, P-by-N matrix.
            y: training sample labels, 1-by-N row vector.

    OUTPUT: w: learned perceptron parameters, (P+1)-by-1 column vector.
            num: number of support vectors

    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    num = 0

    # YOUR CODE HERE
    # Please implement SVM with scipy.optimize. You should be able to implement
    # it within 20 lines of code. The optimization should converge wtih any method
    # that support constrain.
    # begin answer
    x = np.vstack((np.ones((1, N)), X))
    res = fmin_tnc(func, x0=w, args=(x, y, 0.01), approx_grad=False)
    w = res[0]
    pred = np.dot(w, x)
    for p in pred:
        if p >= -1 and p <= 1:
            num += 1     
    # end answer
    return w, num

