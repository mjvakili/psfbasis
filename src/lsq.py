import numpy as np
import ms
import shifter
import shift
import numpy as np
import scipy.optimize as op
from   scipy.linalg import solve
import scipy.linalg as la

class stuff(object):
   
    def __init__(self, data, Q, min_iter=5, max_iter=30, check_iter=5 , tol=1.e-6):

        """ inputs of the code: NxD data matrix;
                                NxD uncertainty matrix;
                                Q = latent dimensionality
        """

        self.N = data.shape[0]           #number of observations
        self.D = data.shape[1]           #input dimensionality, dimension of each observed data point
        self.Q = Q                       #latent dimensionality (note that this also includes the mean, meaning that in initialization
                                         #                       Q-1 pca components are kept!
        self.data = np.atleast_2d(data)  #making sure the data has the right dimension
        self.ivar = ivar                 #variance with the same dimensiona as the data
        
        """ outputs of the code: NxQ amplitude (coefficients) matrix;
                                 QxD basis (eigen vectors) matrix
                                 NX1 flux matrix 
        """
               
        self.A = np.zeros((self.N,self.Q))    #Creating the amplitude matrix
        self.G = np.zeros((self.K,self.D))    #Creating the basis matrix. This matrix contains K eigen-vectors. Each eigen-vector is
                                              # a D-dimensional object!
        self.F = np.zeros((self.N))           #Creating an N-dimensional Flux vector. conatins flux values of N observations.


        """ initialization of FAG by means of normalizing, shifting, and singular-value decomposition"""
        self.initialize()

        """ updating FAG matrix by F-step, A-step, and G-step least-square optimization"""
        self.update(max_iter, check_iter, min_iter, tol)
        
        
         
    def initialize(self):
        """
        initializing the parameters
        """         
        self.dm = np.zeros((self.N,self.D))
          #shifting and normalizing
        """init F"""
        for i in range(self.N):
          
          self.F[i] = np.sum(self.data[i,:])   #flux = sum of pixel intensities
          self.dm[i,:] = shifter.shifter(self.data[i,:])/self.F[i]   #shift each star to its center and normalize it
        
        #initializing the mean. The mean will later be inserted into eigen-vectors.
        mean = np.mean(self.dm, axis=0)

        #subtracting the mean from the shifted/normalized data
        self.dmm = self.dm - self.X
        
        #pca on the mean-subtracted shifted/normalized data
        u , s , vh = la.svd(self.dmm)
        """init A"""
        #amplitudes
        coefficients = u[:, :self.K-1] * s[None, :self.K-1]
        ones = np.ones((self.N,1))
        self.A = np.hstack([ones , coefficients])
        """init G"""
        #eigen basis functions including the mean
        self.G = np.vstack([mean , vh[:self.K-1,:]])

        
"""!!!!!!!!!!!! seems unnecessary for now. svd is already implemented inside initialize()""""     

#    def svd_pca(self, data):

#        U, S, eigvec = np.linalg.svd(data)
#        eigval = S ** 2. / (data.shape[0] - 1.)
#        return eigvec, eigval


    def _e_step(self):
  
        self.latents = np.zeros((self.K, self.N))
        for i in range(self.N):
            ilTl = np.linalg.inv(np.dot(self.G , self.G.T))
            lTx = np.dot(self.G, self.dmm[i])
            self.latents[:, i] = np.dot(ilTl, lTx)
        return self.latents.T 

     
    def update_dmm(self):
        
       
        self.PP = np.zeros((self.D,self.D))
        self.d  = int(np.sqrt(self.D))
        self.FF = np.zeros((self.D))
        #self.dmm *= 0.
 
        for i in range(self.N):
          temp = self.data[i, :].reshape(self.d , self.d)[1:-1,1:-1]
          temp = temp.reshape((self.d-2)*(self.d-2))
          pi = shift.matrix(temp)
          #self.PP += np.dot(pi, pi.T)
          #self.FF += np.dot(pi, self.data[i, :].reshape(1,self.D).T)/(self.F[i])
          self.dmm[i,:] = self.data[i,:]/self.F[i] - np.dot(self.X, pi)
        
        #self.X = solve(self.PP, self.FF)
   
        #temp = self.data[2].reshape(self.d , self.d)[1:-1,1:-1]
        #temp = temp.reshape((self.d-2)*(self.d-2))
        #pi = shift.matrix(temp)
        #r = np.dot(self.X, pi)
        #r = solve(np.dot(pi,pi.T)


    def update_A(self):
       
       self.d = int(np.sqrt(self.D))
       for i in range(self.N):  
         temp = self.data[i, :].reshape(self.d , self.d)[1:-1,1:-1]
         temp = temp.reshape((self.d-2)*(self.d-2))
         pi = shift.matrix(temp)   
         igtilde = np.dot(self.G, pi)
         gg = np.dot(igtilde, igtilde.T)
         gy = np.dot(igtilde, self.dmm[i,:])
         self.A[i,:] = solve(gg, gy)

    
    def update_G(self):
       
       self.d = int(np.sqrt(self.D))
       self.gg = np.zeros_like(self.G)
       for i in range(self.N):
         temp = self.data[i, :].reshape(self.d , self.d)[1:-1,1:-1]
         temp = temp.reshape((self.d-2)*(self.d-2))
         pi = shift.imatrix(temp)   
         aa = np.outer(self.A[i,:].T, self.A[i,:])
         ay = np.zeros((self.K,self.D))
         for j in range(self.D):
         
          #print aa.shape
          ay[:,j] = self.A[i,:].T*self.dmm[i,j]
         self.gg += np.dot(solve(aa, ay),pi)/self.N
         
       self.G = self.gg
       self.orthonormalize()
       for i in range(self.N):
         
         temp = self.data[i, :].reshape(self.d , self.d)[1:-1,1:-1]
         temp = temp.reshape((self.d-2)*(self.d-2))
         pi = shift.imatrix(temp)   
         self.X += np.dot(self.data[i] - self.F[i]*np.dot(self.A[i], self.G), pi)
       self.X /= self.N
    def update_F(self):

       self.d = int(np.sqrt(self.D))
       for i in range(self.N):
         temp = self.data[i, :].reshape(self.d , self.d)[1:-1,1:-1]
         temp = temp.reshape((self.d-2)*(self.d-2))
         pi = shift.matrix(temp)
         imodel = np.dot(self.A[i], self.G) + self.X 
         ismodel = np.dot(imodel , pi)
         #print ismodel.shape
         self.F[i] = np.max(self.data[i])/np.max(ismodel)

    def nll(self):
   
       nll = 0.
       self.d = int(np.sqrt(self.D))
       for i in range(self.N):
         temp = self.data[i, :].reshape(self.d , self.d)[1:-1,1:-1]
         temp = temp.reshape((self.d-2)*(self.d-2))
         pi = shift.matrix(temp)
         imodel = np.dot(self.A[i], self.G) + self.X 
         ismodel = self.F[i]*np.dot(imodel , pi)
         nll += 0.5*np.sum((ismodel - self.data[i,:])**2.)
       return nll

    def orthonormalize(self):
        """
        Ortho_Normalize Gs
        """
        def get_normalization(v):
            return np.sqrt(np.dot(v, v))

        self.G[0] /= get_normalization(self.G[0])
        for i in range(1, self.K):
            for j in range(0, i):
                v = np.dot(self.G[i], self.G[j])
                self.G[i] -=  v * self.G[j]
                    
            self.G[i] /= get_normalization(self.G[i])


    def update(self, max_iter, check_iter, min_iter, tol):
  
        #print 'Starting NLL =', self.nll()
        nll = self.nll()
        for i in range(max_iter):
            self.update_dmm()
            self.update_A()
            self.update_G()
            self.update_F()
            if np.mod(i, check_iter) == 0:
                new_nll = self.nll() 
                print new_nll
            if (min_iter < i):
                print 'Stopping at step %d with NLL:' % i, new_nll
                self.nll = new_nll
                break
            else:
                nll = new_nll
        self.nll = new_nll 
