import numpy as np

def perceptron(X, y):
    '''
    PERCEPTRON Perceptron Learning Algorithm.

       INPUT:  X: training sample features, P-by-N matrix.
               y: training sample labels, 1-by-N row vector.

       OUTPUT: w:    learned perceptron parameters, (P+1)-by-1 column vector.
               iter: number of iterations

    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    iters = 0
    # YOUR CODE HERE
    
    # begin answer
    x = np.vstack((np.ones((1, N)), X))
    lr = 0.1
    while True:
        iters += 1
       
        pred = np.dot(w.T, x)
        dw = np.zeros((P + 1, 1))
        i = pred[0] * y[0] <= 0
        if np.all(i==False) or iters >= 3000:
            break
        
        temp = x[:, i] * y[0, i]
        dw = np.expand_dims(np.sum(temp, axis=1), 1)
        if np.sum(i) / N <= 0.1 and iters>=1000:
            break
        w = w + dw * lr
    # end answer
    
    return w, iters