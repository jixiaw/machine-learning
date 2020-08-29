import numpy as np
from likelihood import likelihood

def posterior(x):
    '''
    POSTERIOR Two Class Posterior Using Bayes Formula
    INPUT:  x, features of different class, C-By-N vector
            C is the number of classes, N is the number of different feature
    OUTPUT: p,  posterior of each class given by each feature, C-By-N matrix
    '''

    C, N = x.shape
    l = likelihood(x)
    total = np.sum(x)
    p = np.zeros((C, N))
    #TODO

    # begin answer
    l = likelihood(x)
    preC = [i / np.sum(x) for i in [np.sum(x[j]) for j in range(C)]]
    preX = [i / np.sum(x) for i in np.sum(x, axis=0)]
    for i in range(C):
        for j in range(N):
            p[i, j] = preC[i] * l[i, j] / preX[j]
    # end answer
    
    return p
