import numpy as np
import ms
import shifter
import shift
import numpy as np
import scipy.optimize as op
from   scipy.linalg import solve
import scipy.linalg as la
import itertools

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
        #self.ivar = ivar                 #variance with the same dimensiona as the data
        
        """ outputs of the code: NxQ amplitude (coefficients) matrix;
                                 QxD basis (eigen vectors) matrix
                                 NX1 flux matrix 
        """
               
        self.A = np.zeros((self.N,self.Q))    #Creating the amplitude matrix
        self.G = np.zeros((self.Q,self.D))    #Creating the basis matrix. This matrix contains K eigen-vectors. Each eigen-vector is
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
          self.dm[i,:] = np.dot(self.data[i,:], shift.imatrix(self.data[i,:]))/self.F[i]   #shift each star to its center and normalize it
        
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
     
     def F_step(self):

        for i in range(self.N):
          Ki = shift.matrix(self.data[i,:])  
          Mi = np.dot(self.A[i,:],np.dot(self.G,Ki)).reshape(self.D,1)
          cov = np.linalg.inv(np.dot(Mi.T, Mi))                
          self.F[i] = np.dot(cov, np.dot(Mi.T, self.data[i,:]))

     def HOGG_A_step(self):
      
        Mf = np.zeros((self.N*self.D , self.N*self.Q))
        for i in range(self.N):
          Ki = shift.matrix(self.data[i,:])
          Mf[i*self.D:(i+1)*self.D , i*self.Q:(i+1)*self.Q] = self.F[i]*np.dot(self.G , Ki).T
          
        cov = np.linalg.inv(np.dot(Mf.T, Mf))
        self.A = np.dot(cov, np.dot(Mf.T, self.data.flatten())).reshape(self.N , self.Q)
   

     def A_step(self):
        
        for i in range(self.N):  
          Ki = shift.matrix(self.data[i,:])   
          Mi = self.F[i]*np.dot(self.G,Ki).T
          cov = np.linalg.inv(np.dot(Mi.T, Mi))
          self.A[i,:] = np.dot(cov, np.dot(Mi.T, self.data[i,:]))

     """HOGG_A_step and A_step are equivalent, but A-step is faster"""


     def G_step(self): 

        Mf = np.zeros((self.D , self.N , self.Q , self.D))
        for i in range(self.N):
          Ki = shift.matrix(self.data[i,:])
          Mf[i] = self.F[i,None,None,None]*self.A[i,None,:,None]*Ki.T[None,:,None,:]
        Tf = Mf.reshape(self.N*self.D, self.Q*self.D)
        cov = np.linalg.inv(np.dot(Tf.T, Tf))
        self.G = np.dot(cov, np.dot(Tf.T, self.data.flatten())).reshape(self.Q , self.D)

     def nll(self):
   
       nll = 0.

       for i in range(self.N):
         Ki = shift.matrix(self.data[i,:])
         #print np.dot(self.A[i,:],self.G).shape
         #print Ki.shape
         model_i = self.F[i]*np.dot(np.dot(self.A[i,:], self.G) , Ki)
         b  = int((self.D)**.5)
         model_square = model_i.reshape(b,b)
         data_square = self.data[i,:].reshape(b,b)
         nll += 0.5*np.sum((model_square - data_square)**2.)
       return nll
     
     def orthonormalize(self):
       
        def get_normalization(v):
            return np.sqrt(np.dot(v, v))

        self.G[0] /= get_normalization(self.G[0])
        for i in range(1, self.Q):
            for j in range(0, i):
                v = np.dot(self.G[i], self.G[j])
                self.G[i] -=  v * self.G[j]
                    
            self.G[i] /= get_normalization(self.G[i])
     

     def update(self, max_iter, check_iter, min_iter, tol):
  
        print 'Starting NLL =', self.nll()
        nll = self.nll()
        for i in range(max_iter):

            np.savetxt("GePrime_10%d.txt"%(i) , np.array(self.G.flatten()) ,fmt='%.12f')
            np.savetxt("AePrime_10%d.txt"%(i) , np.array(self.A.flatten()) ,fmt='%.12f')
            np.savetxt("FePrime_10%d.txt"%(i) , np.array(self.F.flatten()) ,fmt='%.12f')
            
            #self.orthonormalize()
            self.F_step()
            print "NLL after F-step", self.nll()
            self.A_step()
            print "NLL after A-step", self.nll()
            self.G_step()
            print "NLL after G-step", self.nll()
        
            if np.mod(i, check_iter) == 0:
                new_nll =  new_nll = self.nll()
                print 'NLL at step %d is:' % i, new_nll
            if (((nll - new_nll) / nll) < tol) & (min_iter < i):
                print 'Stopping at step %d with NLL:' % i, new_nll
                self.nll = new_nll
                break
            else:
                nll = new_nll
        self.nll = new_nll
