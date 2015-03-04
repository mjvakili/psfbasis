import numpy as np
import ms
import shifter
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
        #update A, G, X, F:
        #self.update()
        
        
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
        #subtracting the mean from the data
        self.dmm = self.data - self.X    
        #pca on the mean-subtracted data
        self.lambdas, evals = self.svd_pca(self.dmm)
        self.G = self.lambdas[:self.K, :]
        self.A  = self._e_step()
        
        self.update(tol, min_iter, max_iter, check_iter)

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

     
    def update_X(self):
        
       
        self.PP = np.zeros((self.D,self.D))
        self.d = int(np.sqrt(self.D))
        self.FF = np.zeros((self.D))
        self.dmm *= 0.
 
        for i in range(self.N):
          temp = self.data[i, :].reshape(self.d , self.d)[1:-1,1:-1]
          temp = temp.reshape((self.d-2)*(self.d-2))
          pi = shift.matrix(temp)
          self.PP += np.dot(pi, pi.T)
          self.FF += np.dot(pi, self.data[i, :].reshape(1,self.D).T)/(self.F[i])
          self.dmm[i,:] = self.data[i,:]- np.dot(self.X, pi)
        
        self.X = solve(self.PP, self.FF)
   
        #temp = self.data[2].reshape(self.d , self.d)[1:-1,1:-1]
        #temp = temp.reshape((self.d-2)*(self.d-2))
        #pi = shift.matrix(temp)
        #r = np.dot(self.X, pi)
        #r = solve(np.dot(pi,pi.T)


   def update_A(self):

       for i in range(self.N):  
         temp = self.data[i, :].reshape(self.d , self.d)[1:-1,1:-1]
         temp = temp.reshape((self.d-2)*(self.d-2))
         pi = shift.matrix(temp)   
         igtilde = np.dot(G, pi)
         gg = np.dot(igtilde, igtilde.T)
         gy = np.dot(igtilde, self.dmm[i,:])
         self.A[i,:] = solve(gg, gy)

    
   def update_G(self):
       
       self.gg = np.zeros_like(self.G)
       for i in range(self.N):
         temp = self.data[i, :].reshape(self.d , self.d)[1:-1,1:-1]
         temp = temp.reshape((self.d-2)*(self.d-2))
         pi = shift.matrix(temp)   
         aa = np.dot(self.A[i,:].T, self.A[i,:])
         ay = np.dot(self.A[i,:].T, self.dmm[i,:])
         self.gg += solve(pi, solve(aa, ay))/self.N
         
       self.G = self.gg

   def update_F(self):

       for i in range(self.N):
         temp = self.data[i, :].reshape(self.d , self.d)[1:-1,1:-1]
         temp = temp.reshape((self.d-2)*(self.d-2))
         pi = shift.matrix(temp)
         imodel = np.dot(self.A[i], self.G) + self.X 
         ismodel = np.dot(imodel , pi)
         self.F[i] = self.data[i]/ismodel

   def nll(self):
   
       nll = 0.
       for i in range(self.N):
         temp = self.data[i, :].reshape(self.d , self.d)[1:-1,1:-1]
         temp = temp.reshape((self.d-2)*(self.d-2))
         pi = shift.matrix(temp)
         imodel = np.dot(self.A[i], self.G) + self.X 
         ismodel = self.F[i]*np.dot(imodel , pi)
         nll += 0.5*np.sum((ismodel - data[i,:])**2.)
       return nll

   def update(self, maxiter, check_iter, min_iter, tol):
  
        print 'Starting NLL =', self.nll()
        nll = self.nll()
        for i in range(max_iter):
            self.update_X()
            self.update_A()
            self.update_G()
            self.update_F()
            if np.mod(i, check_iter) == 0:
                new_nll = self.nll() 
                print 'NLL at step %d is:' % i, new_nll
            if (((nll - new_nll) / nll) < tol) & (min_iter < i):
                print 'Stopping at step %d with NLL:' % i, new_nll
                self.nll = new_nll
                break
            else:
                nll = new_nll
        self.nll = new_nll 
