import numpy as np


class SGD(object):
    
    """
    Parameters
    ----------
    data : N x D array of the data, where N is the number of training samples, and D is the number of features (pixels).
    K    : Dimension of the latent space.
    sigma : Array of variances associated with the data.
    alpha  : Regularization strength
    tol  : Percent tolerance criterion for delta negative log likelihood.
    max_iter   : Maximum number of iterations to run sgd.
    check_iter : Interval number of sgd iterations between convergence checks.
    """
   

    def __init__(self, data, K, sigma , alpha , eta0 , min_iter=10, max_iter=100, check_iter=10, tol=1.e-10):

        """
        `D` : Feature dimensionality
        `N` : Number of samples
        `K` : Latent dimensionality
        `THETA` : Latent transformation matrix, shape K x D
        `X` : Latent variables, shape N x K
        `bg` : flat background sky, shape N x 1
        """

        self.N = data.shape[0]
        self.D = data.shape[1]
        self.K = K 
        self.data = np.atleast_2d(data)
        self.sigma = sigma
        self.alpha = alpha
        self.eta0 = eta0

        # Subtracting the mean and scaling the data
        self.mean = np.sum(self.data / self.sigma, axis=0)
        self.mean /= np.sum(self.sigma, axis=0)
        self.zero_mean = self.data - self.mean
        
        # initialize the latent variables using PCA
        self.THETA, EVALS = self.svd_pca(self.zero_mean)
        self.THETA = self.THETA[:self.K, :]

        # initialize the basis and compute the model
        self.X = np.zeros((self.K, self.N))
        self._update_X()
        self.model()
        

        # go
        self.run_sgd(tol, min_iter, max_iter, check_iter)
        
    def svd_pca(self, data):
        """
        PCA using a singular value decomposition.
        """
        U, S, eigvec = np.linalg.svd(data)
        eigval = S ** 2. / (data.shape[0] - 1.)
        return eigvec, eigval
    
   
    def run_sgd(self, tol, min_iter, max_iter, check_iter):
        """
        Use gradient_descent method to infer the model.
        """
        print .5 * np.sum((self.zero_mean - self.projection) ** 2.)
        nll = 0.5 * np.sum((self.zero_mean - self.projection) ** 2.)
        print 'Starting NLL =', nll
        for i in range(max_iter):
            
            #self.THETA = self.THETA - self.eta0 * np.dot(self.X , self.projection - self.zero_mean)
            self._orthonormalize()
            #print self.THETA.shape
            self._update_X()

            if np.mod(i, check_iter) == 0:
                new_nll =  0.5 * np.sum((self.zero_mean - self.projection) ** 2.  * self.sigma)
           #     print 'NLL at step %d is:' % i, new_nll
           # if (((nll - new_nll) / nll) < tol) & (min_iter < i):
           #     print 'Stopping at step %d with NLL:' % i, new_nll
           #     self.nll = new_nll
           #     break
           # else:
           #     nll = new_nll
        for i in range(max_iter):    
            #self.X = self.X - self.eta0 * np.dot(self.THETA , (self.projection - self.zero_mean).T )
            #print self.X.shape           
            self._update_THETA()


            if np.mod(i, check_iter) == 0:
                new_nll =  0.5 * np.sum((self.zero_mean - self.projection) ** 2.)
                print 'NLL at step %d is:' % i, new_nll
            if (((nll - new_nll) / nll) < tol) & (min_iter < i):
                print 'Stopping at step %d with NLL:' % i, new_nll
                print  np.sum((self.zero_mean - self.projection) ** 2.)
                self.nll = new_nll
                break
            else:
                nll = new_nll
        self.nll = new_nll

    def _update_X(self):
        """
        update the latent variables.
        """

        for i in range(self.N):
            A = np.linalg.inv(np.dot(self.THETA  * self.sigma[i] , self.THETA.T))
            B = np.dot(self.THETA, self.zero_mean[i] * self.sigma[i])
            self.X[:, i] = np.dot(A , B)
    
    def _update_THETA(self):

        """
        update the latent basis.
        """

        for j in range(self.D):
            A = np.linalg.inv(np.dot(self.X * self.sigma[:, j] , self.X.T))
            B = np.dot(self.X, self.zero_mean[:, j] * self.sigma[:, j])
            self.THETA[:, j] = np.dot(A, B)

        self._orthonormalize()
        self.model()


    def _orthonormalize(self):

        """
        Orthonormalize the latent basis
        """

        def get_normalization(v):
            return np.sqrt(np.dot(v, v))

        self.THETA[0] /= get_normalization(self.THETA[0])
        for i in range(1, self.K):
            for j in range(0, i):
                v = np.dot(self.THETA[i], self.THETA[j])
                self.THETA[i] -=  v * self.THETA[j]
                    
            self.THETA[i] /= get_normalization(self.THETA[i])


    def model(self):
        #print self.X.shape
        #print self.THETA.shape
        self.projection = np.dot(self.X.T , self.THETA)

