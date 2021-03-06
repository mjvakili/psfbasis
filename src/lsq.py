import numpy as np
import ms
import shifter
import shift
import numpy as np
from   scipy.linalg import solve
import scipy.linalg as la
from random import randint

class stuff(object):
   
     def __init__(self, data, Q, eps = .25 , eta , min_iter=5, max_iter=30, check_iter=5 , tol=1.e-6):

        """ inputs of the code: NxD data matrix;
                                NxD uncertainty matrix;
                                Q = latent dimensionality
        """

        self.N = data.shape[0]           #number of observations
        self.D = data.shape[1]           #input dimensionality, dimension of each observed data point
        self.Q = Q                       #latent dimensionality (note that this also includes the mean,
                                         #meaning that in initialization Q-1 pca components are kept!
        self.eps = eps                   #Hyper-parameter: coeffcient of L1 reg. term on Basis Functions
        self.eta = eta                   #Hyper-parameter: coeffcient of L2 reg. term on Flux values
        self.data = np.atleast_2d(data)  #making sure the data has the right dimension
        #self.ivar = ivar                #variance with the same dimensiona as the data
        
        """ outputs of the code: NxQ amplitude (coefficients) matrix;
                                 QxD basis (eigen vectors) matrix
                                 NX1 flux matrix 
        """
               
        self.A = np.zeros((self.N,self.Q))    #Creating the amplitude matrix
        self.Z = np.zeros((self.Q,self.D))    #Creating the basis matrix. This matrix contains K eigen-vectors. Each eigen-vector is
                                              # a D-dimensional object!
        self.F = np.zeros((self.N))           #Creating an N-dimensional Flux vector. conatins flux values 
                                              #of N observations.


        """ initialization of FAZ by means of normalizing, shifting, and singular-value decomposition"""
        self.initialize()

        """ updating FAZ matrix by F-step, A-step, and Z-step least-square optimization"""
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
          self.dm[i,:] = np.dot(self.data[i,:], shift.matrix(self.data[i,:]))/self.F[i]   #shift each star 
                                                                                           #to its center and
                                                                                           #normalize it
        
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
        """init Z"""
        #eigen basis functions including the mean
        self.Z = np.vstack([mean , vh[:self.Q-1,:]])

     def F_step(self):

        for i in range(self.N):
          Ki = shift.imatrix(self.data[i,:])
          Mi = np.dot(self.A[i,:],np.dot(self.Z,Ki)).reshape(self.D,1)
          cov = np.linalg.inv(np.dot(Mi.T, Mi))                
          self.F[i] = np.dot(cov, np.dot(Mi.T, self.data[i,:]))

     def A_step_prime(self):
      
        Mf = np.zeros((self.N*self.D , self.N*self.Q))
        for i in range(self.N):
          Ki = shift.imatrix(self.data[i,:])
          Mf[i*self.D:(i+1)*self.D , i*self.Q:(i+1)*self.Q] = self.F[i]*np.dot(self.Z , Ki).T
          
        cov = np.linalg.inv(np.dot(Mf.T, Mf))
        self.A = np.dot(cov, np.dot(Mf.T, self.data.flatten())).reshape(self.N , self.Q)
   

     def A_step(self):
        
        for i in range(self.N):  
          Ki = shift.imatrix(self.data[i,:])   
          Mi = self.F[i]*np.dot(self.Z,Ki).T
          cov = np.linalg.inv(np.dot(Mi.T, Mi))
          self.A[i,:] = np.dot(cov, np.dot(Mi.T, self.data[i,:]))

     """A_step_prime and A_step are equivalent"""


     def Z_step(self): 

        Mf = np.zeros((self.N , self.D , self.Q , self.D))
        for i in range(self.N):
          Ki = shift.imatrix(self.data[i,:])
          Mf[i] = self.F[i,None,None,None]*self.A[i,None,:,None]*Ki.T[None,:,None,:]
        Tf = Mf.reshape(self.N*self.D, self.Q*self.D)
        cov = np.linalg.inv(np.dot(Tf.T, Tf)+.00001*np.eye(self.Q*self.D , self.Q*self.D))
        self.Z = np.dot(cov, np.dot(Tf.T, self.data.flatten())).reshape(self.Q , self.D)


     def SGD_Z_step(self):
        for t in range(1,self.N):
          p = randint(0,self.N-1)
          Kp = shift.imatrix(self.data[p,:])
          modelp = self.data[p,:] - self.F[p]*np.dot(np.dot(self.A[p,:],self.Z),Kp)
          gradp = -2.*self.F[p,None,None,None]*self.A[p,None,:,None]*Kp.T[:,None,:]
          gradp = modelp[:,None,None]*gradp
          Gradp = np.sum(gradp , axis = 0)        
          beta = self.alpha/t
          self.Z = self.Z - beta*Gradp
          
     def grad_Z(self , params , *args):
        
        self.A, self.F = args
        self.Z = params.reshape(self.Q,self.D)
        grad = np.zeros_like(self.Z)
        grad[self.Z>0] =  self.eps
        grad[self.Z<0] = -self.eps
        for p in range(self.N):
         Kp = shift.imatrix(self.data[p,:])
         modelp = self.data[p,:] - self.F[p]*np.dot(np.dot(self.A[p,:],self.Z),Kp)
         gradp = -1.*self.F[p,None,None,None]*self.A[p,None,:,None]*Kp.T[:,None,:]
         gradp = modelp[:,None,None]*gradp
         Gradp = np.sum(gradp , axis = 0) 
         grad += Gradp
        return grad.flatten()
     
     def grad_A(self, params, *args):

        self.Z, self.F = args
        self.A = params.reshape(self.N, self.Q)
        grad = np.zeros_like(self.A)
        for p in range(self.N):
         Kp = shift.imatrix(self.data[p,:])
         modelp = self.data[p,:] - self.F[p]*np.dot(np.dot(self.A[p,:],self.Z),Kp)
         gradp = -1.*self.F[p][None,None]*np.dot(self.Z,Kp)[:,:]
         grad[p,:]   = (gradp*modelp[None,:]).sum(axis=1)
         grad[p,:] *= 0.
        return grad.flatten() 
     
     def grad_F(self, params, *args):

        self.A, self.G = args
        self.F = params
        grad = np.zeros_like(self.F)
        for p in range(self.N):
         Kp = shift.imatrix(self.data[p,:])
         modelp = self.data[p,:] - self.F[p]*np.dot(np.dot(self.A[p,:],self.G),Kp)
         gradp = -1.*np.dot(np.dot(self.A[p,:],self.G),Kp)
         grad[p] = np.sum(modelp*gradp) + 2.*self.eta*self.F[p]
        return grad
        
     def func_A(self, params , *args):
        
        self.Z , self.F = args
        self.A  = params.reshape(self.N, self.Q)
        return self.nll()
          
     def func_Z(self , params, *args):
        self.A, self.F = args
        self.Z = params.reshape(self.Q, self.D)   
        return self.nll()
     
     def func_F(self , params, *args):
        self.A, self.Z = args
        self.F = params
        return self.nll()  

     def bfgs_Z(self):
        x = op.fmin_l_bfgs_b(self.func_Z,x0=self.Z.flatten(), fprime = self.grad_Z,args=(self.A,self.F), approx_grad = False, \
                              bounds = None, m=10, factr=100., pgtol=1e-05, epsilon=1e-04, maxfun=50)
        #print x
        self.Z  = x[0].reshape(self.Q,self.D)
        
     def bfgs_A(self):
        x = op.fmin_l_bfgs_b(self.func_A,x0=self.A.flatten(), fprime = self.grad_A,args=(self.Z,self.F), approx_grad = False, \
                              bounds = None, m=10, factr=100., pgtol=1e-05, epsilon=1e-04, maxfun=50)
        #print x
        self.A  = x[0].reshape(self.N,self.Q)
        
     def bfgs_F(self):
        x = op.fmin_l_bfgs_b(self.func_F,x0=self.F, fprime = self.grad_F,args=(self.A,self.Z), approx_grad = False, \
                              bounds = None, m=10, factr=100., pgtol=1e-05, epsilon=1e-04, maxfun=50)
        #print x
        self.F  = x[0]   
     def svd_A_rotate_A_and_Z(self):

        u_ , s_ , vh_ = la.svd(self.A[:,1:])
        ss_ = np.vstack([np.diag(s_) , np.zeros((self.N - self.Q + 1 , self.Q-1))])
        self.A[:,1:] = np.dot(u_  , ss_)
        self.Z[1:,:] = np.dot(vh_ , self.Z[1:,:])

     def nll(self):
   
       nll = 0.

       for i in range(self.N):
         Ki = shift.imatrix(self.data[i,:])
         model_i = self.F[i]*np.dot(np.dot(self.A[i,:], self.Z) , Ki)
         b  = int((self.D)**.5)
         model_square = model_i.reshape(b,b)
         data_square = self.data[i,:].reshape(b,b)
         nll += 0.5*np.sum((model_square - data_square)**2.)
       return nll + self.esp*np.abs(self.Z).sum() + self.eta*np.sum(self.F**2.)
     
     def orthonormalize(self):
       
        def get_normalization(v):
            return np.sqrt(np.dot(v, v))

        self.Z[0] /= get_normalization(self.Z[0])
        for i in range(1, self.Q):
            for j in range(0, i):
                v = np.dot(self.Z[i], self.Z[j])
                self.Z[i] -=  v * self.Z[j]
                    
            self.Z[i] /= get_normalization(self.Z[i])
     

     def update(self, max_iter, check_iter, min_iter, tol):
  
        print 'Starting NLL =', self.nll()
        nll = self.nll()
        for i in range(max_iter):

            np.savetxt("ZePrime_10%d.txt"%(i) , np.array(self.Z.flatten()) ,fmt='%.12f')
            np.savetxt("AePrime_10%d.txt"%(i) , np.array(self.A.flatten()) ,fmt='%.12f')
            np.savetxt("FePrime_10%d.txt"%(i) , np.array(self.F.flatten()) ,fmt='%.12f')
            
            oldobj = self.nll()
            self.F_step()
            
            obj = self.nll()
            assert (obj < oldobj)or(obj == oldobj)
            print "NLL after F-step", obj
            oldobj = self.nll()
            
            #self.Z_step()
            self.lbfgs_Z()
            obj = self.nll()
            assert (obj < oldobj)or(obj == oldobj)
            print "NLL after Z-step", obj
            oldobj = self.nll()
            
            #self.orthonormalize()
            
            self.lbfgs_A()
            obj = self.nll()
            assert (obj < oldobj)or(obj == oldobj)
            
            print "NLL after A-step , and rotating A and Z", self.nll()
        
        
            if np.mod(i, check_iter) == 0:
                new_nll =  new_nll = self.nll()
                print 'NLL at step %d is:' % i+1, new_nll
            if (((nll - new_nll) / nll) < tol) & (min_iter < i):
                print 'Stopping at step %d with NLL:' % i, new_nll
                self.nll = new_nll
                break
            else:
                nll = new_nll
        self.nll = new_nll
