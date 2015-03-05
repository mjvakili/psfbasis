import numpy as np
import ms
import modelshifter
import shift
import numpy as np
import scipy.optimize as op
from scipy.linalg import solve


class stuff(object):
   
    def __init__(self, data, K, min_iter=5, max_iter=30, check_iter=5 , tol=1.e-6):

        #for i in range(data):
          
          

        self.N = data.shape[0]
        self.D = data.shape[1]
        self.K = K 
        self.data = np.atleast_2d(data)
        #self.ivar = ivar
               
 
        self.A = np.zeros((self.N,self.K))
        self.G = np.zeros((self.K,self.D))
        self.X = np.zeros((self.D))
        self.F = np.zeros((self.N))

        #initialize A, G, X, F:
        self.initialize()
        self.update_A()
        #update A, G, X, F:
        self.update(max_iter, check_iter, min_iter, tol)
        
        
    def initialize(self):
        """
        initializing the parameters
        """         
        self.dm = np.zeros((self.N,self.D))
          #shifting and normalizing
        for i in range(self.N):
          self.F[i] = np.sum(self.data[i,:])
          self.dm[i,:] = shifter.shifter(self.data[i,:])/self.F[i]

        #averaging to get the mean
        self.X = np.sum(self.dm, axis=0)/self.N
        #subtracting the mean from the shifted/normalized data
        self.dmm = self.dm - self.X
        
        """
        for i in range(self.N):
          temp = self.data[i, :].reshape(self.d , self.d)[1:-1,1:-1]
          temp = temp.reshape((self.d-2)*(self.d-2))
          pi = shift.matrix(temp)
          self.dmm[i] = self.data[i] - self.F[i]*np.dot(self.X,pi)""" 
   
        #pca on the mean-subtracted shifted/normalized data
        self.lambdas, evals = self.svd_pca(self.dmm)
        self.G = self.lambdas[:self.K, :]
        self.A  = self._e_step()
        


    def svd_pca(self, data):
        """
        PCA using a singular value decomposition.
        """
        U, S, eigvec = np.linalg.svd(data)
        eigval = S ** 2. / (data.shape[0] - 1.)
        return eigvec, eigval

    def _e_step(self):
  
        self.latents = np.zeros((self.K, self.N))
        for i in range(self.N):
            ilTl = np.linalg.inv(np.dot(self.G , self.G.T))
            lTx = np.dot(self.G, self.dmm[i])
            self.latents[:, i] = np.dot(ilTl, lTx)
        return self.latents.T 

     
    def update_dmm(self):
        
       
        self.PP = np.zeros((self.D,self.D))
        self.d = int(np.sqrt(self.D))
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
