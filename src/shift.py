import ms
import numpy as np
import c3

def matrix(data):
  
  """
  input:
  
  dx = shift along the x-axis
  dy = shift along the y-axis
  M = shape x shape or shape M^.5
  
  output:

  MxM matrix that can be applied to 
  a flattened M-dimensional data vector
  """
  
  shap = int((data.shape[0])**.5)
  #print shap
  #print data.shape
  data = data.reshape(shap,shap)
  center = c3.find_centroid(data)
  
  dx  , dy = center[0] , center[1]
  b = ms.B(shap).T[-1:1,-1:1]
  phx = ms.phi(-dx , shap)[-1:1,-1:1]
  phy = ms.phi(dy , shap)[-1:1,-1:1]
  hx = np.dot(phx , np.linalg.inv(b))
  hy = np.dot(np.linalg.inv(b) , phy)
  hf = np.kron(hx.T, hy)

  return hf.T
def imatrix(data):
  # note to self: cubic spline is not exactly invertible. shift matrix is in most cases a singular matrix
  #  replace it with a cubic spline matrix with negative shift instead.
                  
  """
  input:
  
  dx = shift along the x-axis
  dy = shift along the y-axis
  M = shape x shape or shape M^.5
  
  output:

  MxM matrix that can be applied to 
  a flattened M-dimensional data vector
  """
  
  shap = int((data.shape[0])**.5)
  #print shap
  #print data.shape
  data = data.reshape(shap,shap)
  center = c3.find_centroid(data)
  
  dx  , dy = center[0] , center[1]
  b = ms.B(shap).T[-1:1,-1:1]
  phx = ms.phi(dx , shap)[-1:1,-1:1]
  phy = ms.phi(-dy , shap)[-1:1,-1:1]
  hx = np.dot(phx , np.linalg.inv(b))
  hy = np.dot(np.linalg.inv(b) , phy)
  hf = np.kron(hx.T, hy)

  return hf.T
