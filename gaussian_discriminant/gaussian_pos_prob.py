import numpy as np

def gaussian_pos_prob(X, Mu, Sigma, Phi):
    '''
    GAUSSIAN_POS_PROB Posterior probability of GDA.
    Compute the posterior probability of given N data points X
    using Gaussian Discriminant Analysis where the K gaussian distributions
    are specified by Mu, Sigma and Phi.
    Inputs:
        'X'     - M-by-N numpy array, N data points of dimension M.
        'Mu'    - M-by-K numpy array, mean of K Gaussian distributions.
        'Sigma' - M-by-M-by-K  numpy array (yes, a 3D matrix), variance matrix of
                  K Gaussian distributions.
        'Phi'   - 1-by-K  numpy array, prior of K Gaussian distributions.
    Outputs:
        'p'     - N-by-K  numpy array, posterior probability of N data points
                with in K Gaussian distribsubplots_adjustutions.
    ''' 
    N = X.shape[1]
    K = Phi.shape[0]
    p = np.zeros((N, K))
    #Your code HERE

    # begin answer
    for n in range(N):
        sump = 0
        for k in range(K):
            index = -1/2 * np.dot(np.dot((X[:, n] - Mu[:, k]).T, np.linalg.inv(Sigma[:, :, k])), (X[:, n] - Mu[:, k]))
            # print(up)
            p[n, k] = Phi[k] * 1 / (2 * np.pi * np.linalg.det(Sigma[:, :, k])) * np.exp(index)
            
            sump += p[n, k]
#         print('before: ', p[n, 0])
        p[n, :] = p[n, :] / sump
#         print(sump, ' after: ', p[n, 0])
    # end answer
    
    return p
    