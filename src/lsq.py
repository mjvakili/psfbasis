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
        self.M = data.shape[1]           #input dimensionality, dimension of each observed data point
        self.Q = Q                       #latent dimensionality (note that this also includes the mean, meaning that in initialization
                                         #                       Q-1 pca components are kept!
        self.data = np.atleast_2d(data)  #making sure the data has the right dimension
        #self.ivar = ivar                 #variance with the same dimensiona as the data
        
        """ outputs of the code: NxQ amplitude (coefficients) matrix;
                                 QxD basis (eigen vectors) matrix
                                 NX1 flux matrix 
        """
               
        self.A = np.zeros((self.N,self.Q))    #Creating the amplitude matrix
        self.G = np.zeros((self.Q,self.M))    #Creating the basis matrix. This matrix contains K eigen-vectors. Each eigen-vector is
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
        self.dm = np.zeros((self.N,self.M))
          #shifting and normalizing
        """init F"""
        for i in range(self.N):
          
          self.F[i] = np.sum(self.data[i,:])   #flux = sum of pixel intensities
          self.dm[i,:] = shifter.shifter(self.data[i,:])/self.F[i]   #shift each star to its center and normalize it
        
        #initializing the mean. The mean will later be inserted into eigen-vectors.
        mean = np.mean(self.dm, axis=0)

        #subtracting the mean from the shifted/normalized data
        self.dmm = self.dm - mean
        
        #pca on the mean-subtracted shifted/normalized data
        u , s , vh = la.svd(self.dmm)
        """init A"""
        #amplitudes
        coefficients = u[:, :self.Q-1] * s[None, :self.Q-1]
        ones = np.ones((self.N,1))
        self.A = np.hstack([ones , coefficients])
        """init G"""
        #eigen basis functions including the mean
        self.G = np.vstack([mean , vh[:self.Q-1,:]])
        print self.nll()
        

     def F_step(self):

        for i in range(self.N):
          Ki = shift.matrix(self.data[i,:])  #K_i for star i  
          Mi = np.dot(self.A[i,:],np.dot(self.G,Ki))
          #cov = np.linalg.inv(np.dot(Mi.T, Mi))                 !!!!!!!!!!!!!!!!!!!!!
          cov = (np.dot(Mi.T,Mi))**-1.
          self.F[i] = np.dot(cov, np.dot(Mi.T, self.data[i,:]))
       
     def A_step(self):
        
        for i in range(self.N):  
          Ki = shift.matrix(self.data[i,:])   
          Mi = self.F[i]*np.dot(self.G,Ki).T                    ########SHOULD THERE BE A TRANSPOSE HERE? ASK HOGG AND/OR DFM AND/OR ROSS 
          cov = np.linalg.inv(np.dot(Mi.T, Mi))
          self.A[i,:] = np.dot(cov, np.dot(Mi.T, self.data[i,:]))

     def G_step(self):
       y_temp = np.zeros_like(self.data)
       A_temp = np.zeros_like(self.A)
       G_temp = np.zeros_like(self.G)
       for i in range(self.N):
        A_temp[i,None] = self.F[i]*self.A[i,None]
        for j in range(self.M):
         Ki = shift.imatrix(self.data[i,:])
         y_temp[i, j] = np.dot(self.data[i,:] , Ki)[j]
       
       for j in range(self.M):
         
         cov = np.linalg.inv(np.dot(A_temp.T, A_temp))
         self.G[:,j]= np.dot(cov, np.dot(A_temp.T, y_temp[:,j]))
   


     def nll(self):
   
       nll = 0.

       for i in range(self.N):
         Ki = shift.matrix(self.data[i,:])
      
         model_i = self.F[i]*np.dot(np.dot(self.A[i,:], self.G) , Ki)
         nll += 0.5*np.sum((model_i - self.data[i,:])**2.)
       return nll
     """
     def orthonormalize(self):
       
        def get_normalization(v):
            return np.sqrt(np.dot(v, v))

        self.G[0] /= get_normalization(self.G[0])
        for i in range(1, self.Q):
            for j in range(0, i):
                v = np.dot(self.G[i], self.G[j])
                self.G[i] -=  v * self.G[j]
                    
            self.G[i] /= get_normalization(self.G[i])
     """

     def update(self, max_iter, check_iter, min_iter, tol):
  
        print 'Starting NLL =', self.nll()
        nll = self.nll()
        for i in range(max_iter):
            self.F_step()
            self.A_step()
            self.G_step()
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
